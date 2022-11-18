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
from ast import literal_eval
import numpy as np
import torch
import torch.nn as nn
from ...modelzoo.models.roberta.modeling_roberta import RobertaModel
from ...modelzoo.models.clip.modeling_clip import CLIPVisionModel,CLIPTextModel
from ...modelzoo.models.clip.configuration_clip import CLIPConfig,CLIPTextConfig,CLIPVisionConfig
from ...modelzoo.models.clip.modeling_openclip import OPEN_CLIP
from ...modelzoo.models.clip.modeling_chineseclip import CHINESE_CLIP
from ...utils import losses, get_pretrain_model_path, get_args
from ..application import Application
import math
from torch import Tensor
from typing import List, Optional
import json


class Config_Wrapper:
    def __init__(self,json_data):
        self.json_data=json_data

    def to_json_string(self):
        json_str=json.dumps(self.json_data,ensure_ascii=False)
        return json_str

class CLIPApp(Application):

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, user_defined_parameters={},**kwargs):
        instance=CLIPApp(pretrained_model_name_or_path,user_defined_parameters)
        return instance

    def __init__(self, pretrained_model_name_or_path=None,user_defined_parameters=None, **kwargs):
        super().__init__()
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
            # 先处理配置，再决定后续如何加载权重
            with open(pretrained_model_name_or_path+'/config.json','r') as config_handle:
                self.raw_config=json.load(config_handle)
            
            if ('model_type' in self.raw_config) and (self.raw_config['model_type']=='open_clip'):
                print(self.raw_config)
                self.model_type='open_clip'
                self.config=Config_Wrapper(self.raw_config)# used by trainer
                self.open_clip = OPEN_CLIP(**self.config.json_data)
                checkpoint = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin', map_location=torch.device('cpu'))
                all_model_state_dict = {k.replace('open_clip.',''): v for k, v in checkpoint.items()}
                self.open_clip.load_state_dict(all_model_state_dict)
            elif ('model_type' in self.raw_config) and (self.raw_config['model_type']=='chinese_clip'):
                print(self.raw_config)
                self.model_type='chinese_clip'
                self.config=Config_Wrapper(self.raw_config)# used by trainer
                checkpoint = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin', map_location=torch.device('cpu'))
                all_params={}
                for k, v in checkpoint.items():
                    all_params[k.replace('chinese_clip.','')]=v
                self.chinese_clip = CHINESE_CLIP(**self.config.json_data)
                self.chinese_clip.load_state_dict(all_params,strict=False)
            else:
                self.model_type='huggingface_clip'
                self.text_config = CLIPTextConfig(**self.raw_config['text_config'])
                self.vision_config = CLIPVisionConfig(**self.raw_config['vision_config'])
                self.config=CLIPConfig.from_text_vision_configs(text_config=self.text_config,vision_config=self.vision_config)
                all_model_state_dict = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin', map_location=torch.device('cpu'))
                all_params={}
                text_encoder_params={}
                vision_encoder_params={}
                for key, value in all_model_state_dict.items():
                    # print(key)
                    all_params[key] = value
                    if 'text_encoder.' in key:
                        key=key.replace('text_encoder.','')
                        text_encoder_params[key] = value
                    if 'vision_encoder.' in key:
                        key=key.replace('vision_encoder.','')
                        vision_encoder_params[key] = value
                self.text_encoder=RobertaModel.from_pretrained(pretrained_model_name_or_path,config=self.text_config,state_dict=text_encoder_params)
                self.vision_encoder = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path,config=self.vision_config,state_dict=vision_encoder_params) 
                self.text_shape=all_params['text_projection.weight'].shape
                self.vision_shape=all_params['vision_projection.weight'].shape
                self.text_projection = nn.Linear(self.text_shape[0], self.text_shape[1])
                self.vision_projection= nn.Linear(self.vision_shape[0], self.vision_shape[1])
                self.text_projection.weight=torch.nn.Parameter(all_params['text_projection.weight'])
                self.text_projection.bias=torch.nn.Parameter(all_params['text_projection.bias'])
                self.vision_projection.weight=torch.nn.Parameter(all_params['vision_projection.weight'])
                self.vision_projection.bias=torch.nn.Parameter(all_params['vision_projection.bias'])
                if 'logit_scale' in all_params:
                    self.logit_scale = torch.nn.Parameter(all_params['logit_scale'])
                else:
                    self.logit_scale = nn.Parameter(torch.tensor([np.log(1/0.07)]))

    def forward(self, inputs,feat=None):
        if self.model_type=='open_clip':
            _device=self.open_clip.text_projection.device
            logit_scale = self.open_clip.logit_scale.exp()
        elif self.model_type=='chinese_clip':
            _device=self.chinese_clip.text_projection.device
            logit_scale = self.chinese_clip.logit_scale.exp()
        else:
            _device=self.text_projection.weight.device
            logit_scale = self.logit_scale.exp()
        if 'pixel_values' in inputs:
            inputs['pixel_values']=inputs['pixel_values'].to(_device)
        else:
            inputs['pixel_values']=None
        if 'input_ids' in inputs:
            inputs['input_ids']=inputs['input_ids'].to(_device)
        else:
            inputs['input_ids']=None
        if self.model_type=='open_clip':
            image_embeds, text_embeds = self.open_clip(inputs['pixel_values'], inputs['input_ids']) 
        elif self.model_type=='chinese_clip':
            image_embeds, text_embeds = self.chinese_clip(inputs['pixel_values'], inputs['input_ids'])             
        else:
            image_embeds=None
            text_embeds=None
            if ('input_ids' in inputs) and (inputs['input_ids'] is not None):
                text_outputs = self.text_encoder(input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'].to(_device),
                attention_mask=inputs['attention_mask'].to(_device))
                text_embeds = text_outputs[1]
                text_embeds = self.text_projection(text_embeds)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            if ('pixel_values' in inputs) and (inputs['pixel_values'] is not None):
                vision_outputs = self.vision_encoder(
                    pixel_values=inputs['pixel_values']
                )
                image_embeds = vision_outputs[1].detach()
                image_embeds = self.vision_projection(image_embeds)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        if feat is True:
            return {'image_embeds':image_embeds,'text_embeds':text_embeds}
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        return {'logits_per_text':logits_per_text,'logits_per_image':logits_per_image,'image_embeds':image_embeds,'text_embeds':text_embeds}

    # contrastive loss function, adapted from
    # https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
    def contrastive_loss(self,logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self,similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0    

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        loss = self.clip_loss(forward_outputs['logits_per_text'])
        return {'loss': loss}


