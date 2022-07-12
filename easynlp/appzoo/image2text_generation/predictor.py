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
import albumentations

from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer
from ...utils import io
from PIL import Image
import time
import base64
from io import BytesIO
from torchvision import transforms


class ImageTextGenerationPredictor(Predictor):

    def __init__(self, model_dir, model_cls=None, user_defined_parameters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir
        self.MUTEX = Lock()
        
        model = model_cls(pretrained_model_name_or_path=model_dir, \
            user_defined_parameters=user_defined_parameters).cuda()
        self.model = model.eval()
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.sequence_length = self.text_len + self.img_len
        self.max_generated_num = int(user_defined_parameters.get('max_generated_num', 1))
        self.size = int(user_defined_parameters.get('img_size', 256))
        self.random_crop = bool(user_defined_parameters.get('random_crop', False))

        # image preprocessor
        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {"idx": [], "input_imgs": [], "img_str": []}

        max_seq_length = -1
        for record in in_data:
            if "sequence_length" not in record:
                break
            max_seq_length = max(max_seq_length, record["sequence_length"])
        max_seq_length = self.sequence_length if (max_seq_length == -1) else max_seq_length

        for record in in_data:
            img_str = record[self.first_sequence]

            try:
                self.MUTEX.acquire()
                image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str))).convert("RGB")
                image = np.array(image).astype(np.uint8)
                image = self.preprocessor(image=image)["image"]
                image = (image/127.5 - 1.0).astype(np.float32)
            finally:
                self.MUTEX.release()

            rst["idx"].append(record["idx"]) 
            rst["input_imgs"].append(image)
            rst["img_str"].append(img_str)

        return rst

    def predict(self, in_data):
        idx = in_data["idx"]
        imgs = torch.Tensor(in_data['input_imgs']).cuda()
        imgs = imgs.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        print (imgs.shape)
        _, img_ids = self.model.encode_to_c(imgs)
        print (img_ids.shape)

        gen_token_ids_list = []
        for gen_idx in range(self.max_generated_num):
            gen_token_ids_list.append(self.model.generate(img_ids))
        output = {"idx": idx, "img_ids": img_ids, "gen_token_ids": gen_token_ids_list,
                "img_str": in_data["img_str"]}
        return output

    def postprocess(self, result):
        idx = result["idx"]
        #imgbase64 = result["img_str"]
        token_ids_list = result["gen_token_ids"]

        gen_text_list = []
        for token_ids in token_ids_list:
            gen_text = self.model.decode_to_text(token_ids)         # [B, N], N<=32
            gen_text_list.append(gen_text) 

        new_results = list()
        for r_idx in range(len(idx)):
            new_results.append({
                "idx": idx[r_idx],
                #"imgbase64": imgbase64[r_idx],
                "gen_text": gen_text_list[0][r_idx],
            })
        return new_results

