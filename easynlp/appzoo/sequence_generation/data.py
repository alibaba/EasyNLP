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


import os
import torch
import numpy as np
import json

from ...modelzoo import AutoTokenizer,BertTokenizer
from ...utils import io
from ..dataset import BaseDataset
from .model import T5PegasusTokenizer,sequence_padding


class SequenceGenerationDataset(BaseDataset):
    def __init__(self,
                 data_file,
                 pretrained_model_name_or_path,
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 second_sequence,
                 user_defined_parameters,
                 is_training,
                 *args,
                 **kwargs):
        """
        Args:
            data_file (`str`): The Path of the data
            vocab_file (`str`): The Path of the vocab
            max_seq_length (`int`): Maximum length of truncated sequence
            input_schema (`str`): The column schema of input files, name:type:length
            query_column (`str`): The column name of `query`
            asr_text_column (`str`): The column name of `asr_text`
            review_text_column (`str`): The column name of `review_text`
            video_feature_column (`str`): The column name of `video base64 feature`
            video_feature_column (`str`): The column name of `label`
        """
        super(SequenceGenerationDataset, self).__init__(data_file, input_schema=input_schema,
                                                output_format="dict", *args, **kwargs)
        if user_defined_parameters is not None:
            if type(user_defined_parameters)=='str':
                self.user_defined_parameters=json.loads(user_defined_parameters)
            else:
                self.user_defined_parameters=user_defined_parameters
        else:
            self.user_defined_parameters={}
        
        # add self.max_decoder_length
        self.max_decoder_length = int(self.user_defined_parameters.get("max_decoder_length", 128))

        if not os.path.exists(pretrained_model_name_or_path) and os.path.exists(pretrained_model_name_or_path+'.tgz'):
            os.remove(pretrained_model_name_or_path+'.tgz')
            print('Decompress pretrain model failed, file is removed. Please try again.')
            exit()
            
        if os.path.exists(pretrained_model_name_or_path):
            local_path=pretrained_model_name_or_path
        else:
            raise FileNotFoundError('The provided model path %s does not exist, please check.' % pretrained_model_name_or_path)
        
        # local_path=os.environ['HOME']+'/.easynlp/modelzoo/'+pretrained_model_name_or_path

        self.config_path=local_path+'/config.json'
        with open(self.config_path, 'r') as load_f:
            load_dict = json.load(load_f)
            if 'gpt2' in pretrained_model_name_or_path or ("architectures" not in load_dict):
                self.tokenizer = BertTokenizer(vocab_file=local_path+'/vocab.txt', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
            else:
                if ("architectures" in load_dict) and (load_dict["architectures"][0]=='T5ForConditionalGeneration'):
                    tokenizer_class=T5PegasusTokenizer
                elif ("model_type" in load_dict) and (load_dict["model_type"]=='bart'):
                    tokenizer_class=BertTokenizer
                else:
                    tokenizer_class=AutoTokenizer
                self.tokenizer_class=tokenizer_class  
                self.tokenizer = tokenizer_class.from_pretrained(local_path)
            
            # add self.decoder_only
            self.decoder_only = 'gpt2' in pretrained_model_name_or_path or ("architectures" not in load_dict) or ("architectures" in load_dict and 'bloom' in load_dict.get('model_type', ''))
            self.is_randeng = 'randeng' in pretrained_model_name_or_path or 'randeng' == load_dict.get('model_type', '')
        # add self.is_training
        self.is_training = is_training
        self.max_seq_length = max_seq_length
        self.language = self.user_defined_parameters.get("language", 'zh')

        # Text Features
        assert first_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % first_sequence
        assert second_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % second_sequence
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence

    @property
    def eval_metrics(self):
        return ("bleu", "rouge")

    @property
    def label_enumerate_values(self):
        return []

    def convert_single_row_to_example(self, row):
        """
        Args:
            row (`dict`): a dict of one row
        Returns:
            result (`dict`): a dict of result feature
        """
        if self.first_sequence not in row or self.second_sequence not in row:
            src_text = "[PAD]"
            tgt_text = "[PAD]"
        else:
            src_text = row[self.first_sequence]
            tgt_text = row[self.second_sequence]

        feature=self.generation_convert_single_example_to_feature(src_text, tgt_text, self.tokenizer, max_seq_len=self.max_seq_length)
        return feature

    def batch_fn(self, features):
        """
        Args:
            features (`list`): a list of features produced by `convert_single_row_to_example`
        Returns:
            inputs (`dict`): a dict to model forwarding
        """
        input_ids = sequence_padding(
            [t["input_ids"] for t in features], padding=self.tokenizer.pad_token_id)
        decoder_input_ids = sequence_padding(
            [t["decoder_input_ids"] for t in features], padding=self.tokenizer.pad_token_id)
        attention_mask = sequence_padding(
            [t["attention_mask"] for t in features], padding=0)
        decoder_attention_mask = sequence_padding(
            [t["decoder_attention_mask"] for t in features], padding=0)
        output={
            "input_ids": torch.LongTensor(input_ids),
            "decoder_input_ids": torch.LongTensor(decoder_input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "decoder_attention_mask": torch.LongTensor(decoder_attention_mask),
            "src_text": [t["src_text"] for t in features],
            "label_ids": [t["tgt_text"] for t in features],
        }
        return output
    
    def generation_convert_single_example_to_feature(self, src_text, tgt_text, tokenizer, max_seq_len=128):
        if self.decoder_only and self.is_training:
            input_tokens = tokenizer.encode(src_text)
            output_tokens = tokenizer.encode(tgt_text, max_length=self.max_decoder_length, truncation='only_first')
            if tokenizer.sep_token:
                input_ids = input_tokens[:self.max_seq_length-len(output_tokens[1:])] + output_tokens[1:]
                # input_ids = tokenizer.encode(src_text + ' %s ' % tokenizer.sep_token + tgt_text, max_length=max_seq_len+self.max_decoder_length, truncation='only_first')
            else:
                input_ids = tokenizer.encode('<s>') + input_tokens[:self.max_seq_length-len(output_tokens)-3] + tokenizer.encode('</s>') + output_tokens + tokenizer.encode('</s>')
                # input_ids = tokenizer.encode('<s>'+src_text + '%s' % tokenizer.eos_token + tgt_text+'</s>', max_length=max_seq_len+self.max_decoder_length, truncation='only_first')
        else:
            if self.is_randeng:
                input_ids = tokenizer.encode('[CLS]')[:-1] + tokenizer.encode(src_text, max_length=max_seq_len-1, truncation='only_first')
            else:
                input_ids = tokenizer.encode(src_text, max_length=max_seq_len, truncation='only_first')
        
        if tgt_text is None:
            decoder_input_ids = [101]
        else:
            if self.is_randeng:
                decoder_input_ids = tokenizer.encode('[CLS]')[:-1] + tokenizer.encode(tgt_text, max_length=self.max_decoder_length, truncation='only_first')
            else:
                decoder_input_ids = tokenizer.encode(tgt_text, max_length=self.max_decoder_length, truncation='only_first')
        features = {
            'input_ids': input_ids,
            'decoder_input_ids': decoder_input_ids,
            'attention_mask': [1] * len(input_ids),
            'decoder_attention_mask': [1] * len(decoder_input_ids),
            'src_text': src_text,
            'tgt_text': tgt_text
        }
        # print(features)
        return features
