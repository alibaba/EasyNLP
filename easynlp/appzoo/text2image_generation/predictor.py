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

# from email.mime import base
# import json
import os
# import uuid
import numpy as np
from threading import Lock

import torch

from PIL import Image
import base64
from io import BytesIO
from torchvision import transforms

from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer
from ...utils import io


def save_image(x):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    buffered = BytesIO()
    Image.fromarray(x).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return str(img_str, 'utf-8')

class TextImageGenerationPredictor(Predictor):

    def __init__(self, model_dir, model_cls=None, user_defined_parameters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.MUTEX = Lock()
        
        model = model_cls(pretrained_model_name_or_path=model_dir).cuda()
        self.model = model.eval()
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.sequence_length = self.text_len + self.img_len
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.img_vocab_size = int(user_defined_parameters.get('img_vocab_size', 16384))
        self.max_generated_num = int(user_defined_parameters.get('max_generated_num', 1))

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {"idx": [], "input_ids": []}

        max_seq_length = -1
        for record in in_data:
            if "sequence_length" not in record:
                break
            max_seq_length = max(max_seq_length, record["sequence_length"])
        max_seq_length = self.sequence_length if (max_seq_length == -1) else max_seq_length
        # print("max_seq_length {}".format(max_seq_length))

        for record in in_data:
            text= record[self.first_sequence]
            # text_b = record.get(self.second_sequence, None)
            try:
                self.MUTEX.acquire()
                text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                text_ids = text_ids[: self.text_len]
                n_pad = self.text_len - len(text_ids)
                text_ids += [self.pad_id] * n_pad
                text_ids = np.array(text_ids) + self.img_vocab_size
                
            finally:
                self.MUTEX.release()

            rst["idx"].append(record["idx"]) 
            rst["input_ids"].append(text_ids)

        return rst

    def predict(self, in_data):
        idx = in_data["idx"]
        text_ids = torch.LongTensor(in_data['input_ids']).cuda()
        gen_img_ids_list = []
        for gen_idx in range(self.max_generated_num):
            gen_img_ids_list.append(self.model.generate(text_ids))
        output = {"idx": idx, "text_ids": text_ids, "gen_img_ids": gen_img_ids_list}
        return output

    def postprocess(self, result):
        idx = result["idx"]
        text_ids = result["text_ids"] - self.img_vocab_size    # [B, 32]
        gen_img_ids_list = result["gen_img_ids"]  # [max_generated_num, B, 256]

        bs = len(idx)
        cshape = torch.tensor([bs, 256, 16, 16])
        #gen_imgs = self.model.decode_to_img(gen_img_ids_list, cshape)  # [B, 3, 256, 256]

        new_results = list()
        for b in range(len(idx)):
            text = "".join(self.tokenizer.decode(text_ids[b], skip_special_tokens=True).split(" "))
            gen_img_base64_list = []
            for gen_idx in range(self.max_generated_num):
                gen_imgs = self.model.decode_to_img(gen_img_ids_list[gen_idx], cshape)  # [B, 3, 256, 256]
                gen_img_base64_list.append(save_image(gen_imgs[b]))

            new_results.append({
                "idx": idx[b],
                "text": text,
                "gen_imgbase64": gen_img_base64_list,
            })
        return new_results


class TextImageGenerationKnowlPredictor(Predictor):

    def __init__(self, model_dir, model_cls=None, user_defined_parameters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.MUTEX = Lock()
        
        model = model_cls(pretrained_model_name_or_path=model_dir).cuda()
        self.model = model.eval()
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.sequence_length = self.text_len + self.img_len
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.img_vocab_size = int(user_defined_parameters.get('img_vocab_size', 16384))
        self.max_generated_num = int(user_defined_parameters.get('max_generated_num', 1))
        self.entity_num = int(user_defined_parameters.get('entity_num', 10))
        self.entity_emb_path = user_defined_parameters.get('entity_emb_path')
        self.embed = torch.nn.Embedding.from_pretrained(torch.load(self.entity_emb_path)).cuda()

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {"idx": [], "input_ids": [], "words_mat": []}

        max_seq_length = -1
        for record in in_data:
            if "sequence_length" not in record:
                break
            max_seq_length = max(max_seq_length, record["sequence_length"])
        max_seq_length = self.sequence_length if (max_seq_length == -1) else max_seq_length
        # print("max_seq_length {}".format(max_seq_length))

        for record in in_data:
            text= record[self.first_sequence]
            lex_id = record['lex_id']
            pos_s = record['pos_s']
            pos_e = record['pos_e']
            token_len = int(record['token_len'])
            try:
                # preprocess text
                self.MUTEX.acquire()
                text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
                text_ids = text_ids[: self.text_len]
                n_pad = self.text_len - len(text_ids)
                text_ids += [self.pad_id] * n_pad
                text_ids = np.array(text_ids) + self.img_vocab_size

                # preprocess word_matrix
                words_mat = np.zeros([self.entity_num, self.text_len], dtype=np.int)
                if len(lex_id) > 0:
                    ents = lex_id.split(' ')[:self.entity_num]
                    pos_s = [int(x) for x in pos_s.split(' ')]
                    pos_e = [int(x) for x in pos_e.split(' ')]
                    ent_pos_s = pos_s[token_len:token_len+self.entity_num]
                    ent_pos_e = pos_e[token_len:token_len+self.entity_num]

                    for i, ent in enumerate(ents):
                        words_mat[i, ent_pos_s[i]:ent_pos_e[i]+1] = ent
                
            finally:
                self.MUTEX.release()

            rst["idx"].append(record["idx"]) 
            rst["input_ids"].append(text_ids)
            rst["words_mat"].append(words_mat)

        return rst

    def predict(self, in_data):
        idx = in_data["idx"]
        text_ids = torch.LongTensor(in_data['input_ids']).cuda()
        words_mat = torch.LongTensor(in_data['words_mat']).cuda()
        words_emb = self.embed(words_mat)
        gen_img_ids_list = []
        for gen_idx in range(self.max_generated_num):
            gen_img_ids_list.append(self.model.generate(text_ids, words_emb))
        output = {"idx": idx, "text_ids": text_ids, "gen_img_ids": gen_img_ids_list}
        return output

    def postprocess(self, result):
        idx = result["idx"]
        text_ids = result["text_ids"] - self.img_vocab_size    # [B, 32]
        gen_img_ids_list = result["gen_img_ids"]  # [max_generated_num, B, 256]

        bs = len(idx)
        cshape = torch.tensor([bs, 256, 16, 16])
        #gen_imgs = self.model.decode_to_img(gen_img_ids_list, cshape)  # [B, 3, 256, 256]

        new_results = list()
        for b in range(len(idx)):
            text = "".join(self.tokenizer.decode(text_ids[b], skip_special_tokens=True).split(" "))
            gen_img_base64_list = []
            for gen_idx in range(self.max_generated_num):
                gen_imgs = self.model.decode_to_img(gen_img_ids_list[gen_idx], cshape)  # [B, 3, 256, 256]
                gen_img_base64_list.append(save_image(gen_imgs[b]))

            new_results.append({
                "idx": idx[b],
                "text": text,
                "gen_imgbase64": gen_img_base64_list,
            })
        return new_results

