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

#from email.mime import base
#import json
import os
#import uuid
import numpy as np
from threading import Lock

from PIL import Image
import time
import base64
import json
from io import BytesIO
from torchvision import transforms

import torch
import albumentations

from ...core.predictor import Predictor, get_model_predictor
# from ...modelzoo import AutoTokenizer
from .clip import _transform as build_clip_image_transform
from ...utils import io



class CLIPGPTFrameTextGenerationPredictor(Predictor):

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
        self.img_size = int(user_defined_parameters.get('img_size', 224))

        # image sequence length and size validation
        assert self.img_size == self.model.first_stage_model.input_resolution, \
            'img_size is not equal to the input_resolution of vit'

        patch_size = self.model.first_stage_model.patch_size
        assert self.img_len == ((self.img_size/patch_size) * (self.img_size/patch_size)), \
            'the value of \'img_len\' must be equal to the square of vit.input_resolution/vit.patch_size'

        # image preprocessor
        self.preprocessor = build_clip_image_transform(self.img_size)

        # frame length
        self.frame_num = int(user_defined_parameters.get('frame_num', '3'))
        assert self.frame_num <= self.img_len, 'the frame_num should not larger than img_len'

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
            img_str_list = json.loads(record[self.first_sequence])

            image_list = []
            for img_str in img_str_list:
                try:
                    self.MUTEX.acquire()
                    image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str)))
                    image = self.preprocessor(image)
                    image_list.append(image.tolist())   #image = image.tolist()

                finally:
                    self.MUTEX.release()

            rst["idx"].append(record["idx"]) 
            rst["input_imgs"].append(image_list)
            rst["img_str"].append(img_str)

        return rst

    def predict(self, in_data):
        idx = in_data["idx"]
        imgs = torch.Tensor(in_data['input_imgs']).cuda() # [B, frame_num, 3, 224, 224]
        assert (imgs.shape[1] == self.frame_num and imgs.shape[2] == 3 and imgs.shape[3] == self.img_size and \
            imgs.shape[4] == self.img_size), 'invalid image shape'

        # encode image to visual features
        imgs = imgs.view(-1, 3, self.img_size, self.img_size)  # [B * frame_num, 3, img_size=224, img_size=224]
        image_embedding_features = self.model.first_stage_model(imgs) # [8 * frame_num, 768 or 1024]
        n_embed_dim = image_embedding_features.shape[-1]
        image_embedding_features = image_embedding_features.view(-1, self.frame_num, n_embed_dim)  # [8, frame_num, 768 or 1024]

        gen_token_ids_list = []
        for gen_idx in range(self.max_generated_num):
            gen_token_ids_list.append(self.model.generate(prefix_token_inputs = None, \
                predix_image_embedding = image_embedding_features))
        output = {"idx": idx, "gen_token_ids": gen_token_ids_list,
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
            gen_multiple_text_list = []
            for gen_idx in range(self.max_generated_num):
                gen_multiple_text_list.append(gen_text_list[gen_idx][r_idx])

            new_results.append({
                "idx": idx[r_idx],
                #"imgbase64": imgbase64[r_idx],
                "gen_text": gen_multiple_text_list,
            })
        return new_results

