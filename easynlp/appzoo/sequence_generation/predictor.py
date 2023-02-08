# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import uuid
import numpy as np
import torch
from ...core.predictor import Predictor, get_model_predictor
from ...utils import io
from ...modelzoo import AutoTokenizer,BertTokenizer,GPT2LMHeadModel
from .model import SequenceGeneration,T5PegasusTokenizer,sequence_padding
from ...utils.global_vars import get_args, set_global_variables, get_tensorboard_writer, parse_user_defined_parameters
from threading import Lock

def generation_convert_single_example_to_feature(src_text, tgt_text, tokenizer, max_seq_len=128):
    input_ids = tokenizer.encode(src_text, max_length=max_seq_len, truncation='only_first')
    if tgt_text is None:
        decoder_input_ids = [101]
    else:
        decoder_input_ids = tokenizer.encode(tgt_text, max_length=max_seq_len, truncation='only_first')
    features = {
        'input_ids': input_ids,
        'decoder_input_ids': decoder_input_ids,
        'attention_mask': [1] * len(input_ids),
        'decoder_attention_mask': [1] * len(decoder_input_ids),
        'src_text': src_text,
        'tgt_text': tgt_text
    }
    return features

class SequenceGenerationPredictor(Predictor):
    def __init__(self, model_dir, model_cls, user_defined_parameters,**kwargs):
        super(SequenceGenerationPredictor, self).__init__(kwargs)
        self.user_defined_parameters=user_defined_parameters
        if os.path.exists(model_dir):
            local_path=model_dir
        else:
            raise FileNotFoundError('The provided model path %s does not exist, please check.' % model_dir)
        self.model_dir=local_path
        config_path=local_path+'/config.json'
        with open(config_path,'r') as load_f:
            load_dict = json.load(load_f)
            self.is_gpt2='gpt2' in model_dir or ("architectures" not in load_dict)
            self.decoder_only = 'gpt2' in model_dir or ("architectures" not in load_dict) or ("architectures" in load_dict and 'bloom' in load_dict.get('model_type', ''))
            self.is_randeng = 'randeng' == load_dict.get('model_type', '')

        if self.is_gpt2:
            self.tokenizer = BertTokenizer(vocab_file=local_path+'/vocab.txt', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
        else:
            with open(config_path,'r') as load_f:
                load_dict = json.load(load_f)
                if ("model_type" in load_dict) and (load_dict["model_type"]=='mt5'):
                    tokenizer_class=T5PegasusTokenizer
                elif ("model_type" in load_dict) and (load_dict["model_type"]=='bart'):
                    tokenizer_class=BertTokenizer
                else:
                    tokenizer_class=AutoTokenizer
                self.tokenizer_class=tokenizer_class  
            self.tokenizer = tokenizer_class.from_pretrained(self.model_dir)

        device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        self.model = model_cls(pretrained_model_name_or_path=self.model_dir,user_defined_parameters=self.user_defined_parameters)
        self.model.to(device)
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        self.MUTEX = Lock()
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.max_encoder_length = kwargs.get('max_encoder_length', int(self.user_defined_parameters.get("max_encoder_length", 512)))
        self.min_decoder_length = int(self.user_defined_parameters.get("min_decoder_length", 8))
        self.max_decoder_length = int(self.user_defined_parameters.get("max_decoder_length", 128))
        self.no_repeat_ngram_size = int(self.user_defined_parameters.get("no_repeat_ngram_size", 2))
        self.num_beams = int(self.user_defined_parameters.get("num_beams", 5))
        self.num_return_sequences = int(self.user_defined_parameters.get("num_return_sequences", 5))
        self.num_beam_groups = int(self.user_defined_parameters.get("num_beam_groups", 0)) or None
        self.diversity_penalty = float(self.user_defined_parameters.get("diversity_penalty", 0)) or (1.0 if self.num_beam_groups else None)

    def preprocess(self, in_data):
        """
        Args:
            in_data (`list`): a list of row dicts
        Returns:
            features (`dict`): a dict of batched features
        """
        if isinstance(in_data, dict):
            in_data = [in_data]

        rst = {
            'input_ids': list(),
            'attention_mask': list(),
        }

        for record in in_data:
            text = record[self.first_sequence]
            try:
                self.MUTEX.acquire()
                feature = generation_convert_single_example_to_feature(
                text, None, tokenizer=self.tokenizer, max_seq_len=self.max_encoder_length)
            finally:
                self.MUTEX.release()
            for key in rst:
                rst[key].append(feature[key])
        return rst

    def predict(self, in_data):
        """
        Args:
            in_data (`dict`): a dict of batched features produced by self.preprocess
        Returns:
            result (`dict`): a dict of result tensors (`np.array`) by model.forward
        """
        input_ids = torch.LongTensor(sequence_padding(
            in_data["input_ids"], padding=self.tokenizer.pad_token_id))
        attention_mask = torch.LongTensor(sequence_padding(
            in_data["attention_mask"], padding=0))
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        if self.decoder_only:
            self.input_len = input_ids.size(1)
            max_decoder_length = self.max_decoder_length + self.input_len
        else:
            max_decoder_length = self.max_decoder_length
        if self.is_randeng:
            eos_token_id = self.tokenizer.eos_token_id
        else:
            eos_token_id = self.tokenizer.sep_token_id
        tmp = self.model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  num_beams=self.num_beams,
                                  min_length=self.min_decoder_length,
                                  max_length=max_decoder_length,
                                  early_stopping=True,
                                  no_repeat_ngram_size=self.no_repeat_ngram_size,
                                  num_return_sequences=self.num_beams,
                                  decoder_start_token_id=self.tokenizer.cls_token_id,
                                  eos_token_id=eos_token_id, 
                                  num_beam_groups=self.num_beam_groups, 
                                  diversity_penalty=self.diversity_penalty)
        rst = {
            "beam_list": list()
        }
        for b in range(len(in_data["input_ids"])):
            rst["beam_list"].append(tmp[b * self.num_beams: (b + 1) * self.num_beams])
        return rst

    def postprocess(self, result):
        """
        Args:
            result (`dict`): a dict of result tensors (`np.array`) produced by self.predict
        Returns:
            result (`list`): a list of dict of result for writing to file
        """
        rst = list()
        
        for b in range(len(result["beam_list"])):
            beams = result["beam_list"][b]
            if self.decoder_only:
                pred_tokens = [self.tokenizer.decode(t[self.input_len:], skip_special_tokens=True) for t in beams]
            else:
                pred_tokens = [self.tokenizer.decode(t[1:], skip_special_tokens=True) for t in beams]
            pred_tokens = [''.join(i.split()) for i in pred_tokens] if self.model._is_zh else pred_tokens
            pred_tokens = pred_tokens[:self.num_return_sequences]
            rst.append({
                "predictions": pred_tokens[0],
                "beams": "||".join(pred_tokens)
            })
        return rst
