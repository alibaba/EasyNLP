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

from email.mime import base
import json
import os
import uuid
import numpy as np
from threading import Lock

import torch

from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer
from ...utils import io
from PIL import Image
import time
import base64
from io import BytesIO
from torchvision import transforms


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

    def preprocess(self, in_data):
        # print("in_data = ", in_data)
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
        gen_img_ids = self.model.generate(text_ids)
        output = {"idx": idx, "text_ids": text_ids, "gen_img_ids": gen_img_ids}
        return output

    def postprocess(self, result):
        idx = result["idx"]
        text_ids = result["text_ids"] - self.img_vocab_size    # [B, 32]
        gen_img_ids = result["gen_img_ids"]
        bs = len(idx)
        cshape = torch.tensor([bs, 256, 16, 16])
        gen_imgs = self.model.decode_to_img(gen_img_ids, cshape)  # [B, 3, 256, 256]

        new_results = list()
        for b in range(len(idx)):
            text = self.tokenizer.decode(text_ids[b], skip_special_tokens=True)
            gen_image = tensor2img(gen_imgs[b])
            gen_img_base64 = img2base64(gen_image)
            new_results.append({
                "idx": idx[b],
                "text": text,
                "gen_imgbase64": gen_img_base64,
            })
        return new_results

def tensor2img(tensor):
    topil = transforms.ToPILImage('RGB')
    img = topil(tensor)
    return img

def img2base64(img):
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format if img.format else 'PNG')
    byte_data = img_buffer.getvalue()
    base64_str = str(base64.b64encode(byte_data), 'utf-8')
    return base64_str  
