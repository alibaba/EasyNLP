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
from ...utils import io
from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer
from easynlp.utils import get_pretrain_model_path
from PIL import Image
import base64
from io import BytesIO
import numpy as np
from .model import MultiModal
from .data import _center_crop, _resize, _to_numpy_array, _normalize

class MultiModalPredictor(Predictor):
    def __init__(self, model_dir, model_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = get_pretrain_model_path(model_dir)
        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.multi_modal=MultiModal.from_pretrained(model_dir, *args, **kwargs)
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.second_sequence = kwargs.pop("second_sequence", "second_sequence")
        self.sequence_length = kwargs.pop("sequence_length", 128)
        self.do_resize=True
        self.size=224
        self.resample=Image.BICUBIC
        self.do_center_crop=True
        self.crop_size=224
        self.do_normalize=True
        self.batch_cnt=0

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
        for record in in_data:
            first_sequence_content = record.get(self.first_sequence, None)
            if self.first_sequence=='text':
                tked = self.tokenizer(first_sequence_content,
                            padding='max_length',
                            truncation=True,
                            max_length=max_seq_length, return_tensors="pt")
                record["input_ids"]=tked["input_ids"]
                record["token_type_ids"]=tked["token_type_ids"]
                record["attention_mask"]=tked["attention_mask"]
            if self.first_sequence=='image':
                one_image= Image.open(BytesIO(base64.urlsafe_b64decode(first_sequence_content)))
                images=[one_image]
                # transformations (resizing + center cropping + normalization)
                if self.do_resize and self.size is not None and self.resample is not None:
                    images = [_resize(image=image, size=self.size, resample=self.resample) for image in images]
                if self.do_center_crop and self.crop_size is not None:
                    images = [_center_crop(image, self.crop_size) for image in images]
                if self.do_normalize:
                    images = [_normalize(image=image) for image in images]
                images=torch.tensor(images)
                record['pixel_values']=images
        return in_data

    def predict(self, in_data):
        if 'pixel_values' in in_data[0]:
            output={'pixel_values':[]}
            for one_data in in_data:
                output['pixel_values'].append(one_data['pixel_values'])
            output['pixel_values']=torch.cat(output['pixel_values'],dim=0)
        if 'input_ids' in in_data[0]:
            output={'input_ids':[],'token_type_ids':[],'attention_mask':[]}
            for one_data in in_data:
                output['input_ids'].append(one_data['input_ids'])
                output['token_type_ids'].append(one_data['token_type_ids'])
                output['attention_mask'].append(one_data['attention_mask'])
            output['input_ids']=torch.cat(output['input_ids'],dim=0)
            output['token_type_ids']=torch.cat(output['token_type_ids'],dim=0)
            output['attention_mask']=torch.cat(output['attention_mask'],dim=0)
        forward_result=self.multi_modal(output,feat=True)
        return forward_result

    def postprocess(self, result):
        if result['image_embeds'] is not None:
            image_embeds_arr=result['image_embeds'].detach().numpy()
            _tmp_image=[]
            for one_emb in image_embeds_arr:
                _tmp_image.append({'image_feat':'\t'.join([str(x) for x in one_emb])})
            return _tmp_image

        if result['text_embeds'] is not None:
            text_embeds_arr=result['text_embeds'].detach().numpy()
            _tmp_text=[]
            for one_emb in text_embeds_arr:
                _tmp_text.append({'text_feat':'\t'.join([str(x) for x in one_emb])})
            return _tmp_text