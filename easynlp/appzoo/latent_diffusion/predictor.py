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

import uuid
import os
import torch
import json
from ...utils import io
from ...core.predictor import Predictor, get_model_predictor
from easynlp.utils import get_pretrain_model_path
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from einops import rearrange
from .model import LatentDiffusion

class LatentDiffusionPredictor(Predictor):
    def __init__(self, model_dir, model_cls=None, args={},user_defined_parameters={},**kwargs):
        super().__init__()
        try:
            self.args=vars(args)
        except:
            self.args=args
        self.user_defined_parameters=user_defined_parameters
        if 'pipeline_params' in args:
            self.user_defined_parameters=args['pipeline_params']
        model_dir = get_pretrain_model_path(model_dir)
        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir
        self.first_sequence = self.args.pop("first_sequence", "first_sequence")
        self.second_sequence = self.args.pop("second_sequence", "second_sequence")
        self.sequence_length = self.args.pop("sequence_length", 128)
        self.ld=model_cls.from_pretrained(model_dir,self.args, self.user_defined_parameters)
        self.gen_cnt=0
        
    def reset(self,n_samples,sample_steps):
        self.ld.reset_params(n_samples,sample_steps)

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")
        if not isinstance(in_data, list):
            in_data = [in_data]
        for_next=[]
        max_seq_length = -1
        for record in in_data:
            if not "sequence_length" in record:
                break
            max_seq_length = max(max_seq_length, record["sequence_length"])
        max_seq_length = self.sequence_length if (max_seq_length == -1) else max_seq_length
        return in_data

    def predict(self, in_data):
        forward_result=self.ld.forward_predict(in_data)
        return forward_result

    def postprocess(self, result):
        all_result=list()
        if self.ld.write_image is True:
            os.makedirs(self.ld.image_prefix, exist_ok=True)
            for idx1,one in enumerate(result):
                cur_path=self.ld.image_prefix+str(self.gen_cnt)
                self.gen_cnt+=1
                os.makedirs(cur_path, exist_ok=True)
                with open(os.path.join(cur_path, "prompt.txt"),'w') as prompt:
                    prompt.write(one['text']+'\n')
                for idx2,x_sample in enumerate(one['image_tensor']):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(cur_path, f"{idx2:04}.png"))
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='png')
                    byte_data = img_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data)
                    all_result.append({'idx':one["idx"],'text':one["text"],'gen_imgbase64':base64_str,'image_tensor':one['image_tensor']})
        else:
            for idx1,one in enumerate(result):
                for idx2,x_sample in enumerate(one['image_tensor']):
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img_buffer = BytesIO()
                    img.save(img_buffer, format='png')
                    byte_data = img_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data)
                    all_result.append({'idx':one["idx"],'text':one["text"],'gen_imgbase64':base64_str,'image_tensor':one['image_tensor']})
        return all_result
