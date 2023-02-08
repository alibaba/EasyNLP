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

class Text2VideoRetrieval(Application):

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, user_defined_parameters={},**kwargs):
        instance=Text2VideoRetrieval(pretrained_model_name_or_path,user_defined_parameters)
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
           
    def forward(self, inputs,feat=None):
        if self.model_type=='open_clip':
            _device=self.open_clip.text_projection.device
            logit_scale = self.open_clip.logit_scale.exp()
        
        if 'pixel_values' in inputs:
            inputs['pixel_values']=inputs['pixel_values'].to(_device)
            inputs['video_masks']=inputs['video_masks'].to(_device)
            B,T,C,H,W = inputs['pixel_values'].shape
            inputs['pixel_values'] = inputs['pixel_values'].view(B*T,C,H,W)
        else:
            inputs['pixel_values']=None
        if 'input_ids' in inputs:
            inputs['input_ids']=inputs['input_ids'].to(_device)
        else:
            inputs['input_ids']=None
        if self.model_type=='open_clip':
            video_embeds=None
            text_embeds=None
            if inputs['pixel_values'] is not None:
                image_features = self.open_clip.encode_image(inputs['pixel_values'])
                image_features = image_features.view(B,T,-1)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                video_features = self._mean_pooling_for_similarity_visual(image_features, inputs['video_masks'])
                video_embeds = video_features / video_features.norm(dim=-1, keepdim=True)
            if inputs['input_ids'] is not None:
                text_features = self.open_clip.encode_text(inputs['input_ids'])
                text_embeds = text_features / text_features.norm(dim=-1, keepdim=True)
        
        if feat is True:
            return {'video_embeds':video_embeds,'text_embeds':text_embeds}
        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, video_embeds.t()) * logit_scale
        logits_per_video = logits_per_text.T
        return {'logits_per_text':logits_per_text,'logits_per_video':logits_per_video,'video_embeds':video_embeds,'text_embeds':text_embeds}
    
    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

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


