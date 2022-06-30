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
import torch
import torch.nn as nn
from ...modelzoo.models.roberta.modeling_roberta import RobertaModel
from ...modelzoo.models.clip.modeling_clip import CLIPVisionModel
from ...modelzoo.models.clip.configuration_clip import CLIPConfig,CLIPTextConfig,CLIPVisionConfig
from ...utils import losses, get_pretrain_model_path, get_args
from ..application import Application
import math
from torch import Tensor
from typing import List, Optional
import json

class MultiModal(Application):

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, user_defined_parameters={},**kwargs):
        instance=MultiModal(pretrained_model_name_or_path,user_defined_parameters)
        return instance

    def __init__(self, pretrained_model_name_or_path=None,user_defined_parameters=None, **kwargs):
        super().__init__()
        if pretrained_model_name_or_path is not None:
            if not hasattr(user_defined_parameters,'mode'):
                user_defined_parameters['mode']='finetune'
            self._state_dict=self.state_dict
            self.args = get_args()
            # 预训练模式从odps读取预生成的clip feature,finetune模式需要即时生成clip feature
            pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
            all_model_state_dict = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin',
                                            map_location=torch.device('cpu'))
            all_params={}
            roberta_params={}
            clip_vit_params={}
            for key, value in all_model_state_dict.items():
                all_params[key] = value
                if 'roberta.' in key:
                    key=key.replace('roberta.','')
                    roberta_params[key] = value
                if 'vit.' in key:
                    key=key.replace('vit.','')
                    clip_vit_params[key] = value
            with open(pretrained_model_name_or_path+'/config.json','r') as config_handle:
                self.raw_config=json.load(config_handle)
            self.text_config = CLIPTextConfig(**self.raw_config['text_config'])
            self.vision_config = CLIPVisionConfig(**self.raw_config['vision_config'])
            self.config=CLIPConfig.from_text_vision_configs(text_config=self.text_config,vision_config=self.vision_config)
            self.roberta=RobertaModel.from_pretrained(pretrained_model_name_or_path,config=self.text_config,state_dict=roberta_params)
            self.roberta.train()

            self.text_shape=all_params['text_projection.weight'].shape
            self.visual_shape=all_params['visual_projection.weight'].shape
            self.text_projection = nn.Linear(self.text_shape[0], self.text_shape[1], bias=False)
            self.visual_projection= nn.Linear(self.visual_shape[0], self.visual_shape[1], bias=False)
            self.text_projection.weight=torch.nn.Parameter(all_params['text_projection.weight'])
            self.visual_projection.weight=torch.nn.Parameter(all_params['visual_projection.weight'])
            self.logit_scale = torch.nn.Parameter(all_params['logit_scale'])

            if user_defined_parameters['mode']=='finetune':
                self.vit = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path,config=self.vision_config,state_dict=clip_vit_params) 
                for param in self.vit.parameters():
                    param.requires_grad = False
                self.visual_projection.requires_grad = False

    def forward(self, inputs,feat=None):
        if self.roberta.device!=self.text_projection.weight.device:
            self.roberta=self.roberta.to(self.text_projection.weight.device)
        image_embeds=None
        text_embeds=None
        if 'input_ids' in inputs:
            text_outputs = self.roberta(input_ids=inputs['input_ids'].to(self.text_projection.weight.device),
            token_type_ids=inputs['token_type_ids'].to(self.text_projection.weight.device),
            attention_mask=inputs['attention_mask'].to(self.text_projection.weight.device))
            text_embeds = text_outputs[1]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        if 'pixel_values' in inputs:
            vision_outputs = self.vit(
                pixel_values=inputs['pixel_values'].to(self.text_projection.weight.device)
            )
            image_embeds = vision_outputs[1].detach()
            image_embeds = self.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        if feat is True:
            return {'image_embeds':image_embeds,'text_embeds':text_embeds}
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
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


