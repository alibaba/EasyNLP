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
from ...modelzoo.models.clip.modeling_clip import CLIPModel
from ...modelzoo import AutoConfig
from ...utils import losses, get_pretrain_model_path, get_args
from ..application import Application
import math
from torch import Tensor
from typing import List, Optional

class MultiModal(Application):

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, user_defined_parameters={},**kwargs):
        instance=MultiModal(None,user_defined_parameters)
        instance.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        instance.args = get_args()
        pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
        pretrained_model_name_or_path_vision = get_pretrain_model_path(user_defined_parameters['pretrain_model_name_or_path_vision'])
        if not hasattr(user_defined_parameters,'mode'):
            user_defined_parameters['mode']='finetune'
        if user_defined_parameters['mode']=='finetune':
            instance.clip = CLIPModel.from_pretrained(pretrained_model_name_or_path_vision) 
            for param in instance.clip.parameters():
                param.requires_grad = False
        model_state_dict = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin',
                                            map_location=torch.device('cpu'))
        instance.logit_scale = nn.Parameter(torch.ones([]) * model_state_dict['logit_scale'].item())
        instance.text_projection = nn.Linear(1024, 768, bias=False)
        instance.text_projection.weight.data=model_state_dict['text_projection.weight']
        roberta_params={}
        for key, value in model_state_dict.items():
            if 'bertish.' in key:
                key=key.replace('bertish.','')
                roberta_params[key] = value
        instance.bertish=RobertaModel.from_pretrained(pretrained_model_name_or_path,state_dict=roberta_params)
        instance.bertish.train()
        return instance

    def pseudo_state_dict(self):
        verbose_dict=self._state_dict()
        not_clip_params={}
        for key, value in verbose_dict.items():
            if 'clip.' not in key:
                not_clip_params[key] = value
        return not_clip_params

    def __init__(self, pretrained_model_name_or_path=None,user_defined_parameters=None, **kwargs):
        super().__init__()
        if pretrained_model_name_or_path is not None:
            self._state_dict=self.state_dict
            self.state_dict=self.pseudo_state_dict
            self.args = get_args()
            # 预训练模式从odps读取预生成的clip feature,finetune模式需要即时生成clip feature
            pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
            pretrained_model_name_or_path_vision = get_pretrain_model_path(user_defined_parameters['pretrain_model_name_or_path_vision'])
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            roberta_model_state_dict = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin',
                                            map_location=torch.device('cpu'))
            roberta_params={}
            for key, value in roberta_model_state_dict.items():
                if ('bert.' in key) or ('bertish.' in key):
                    key=key.replace('bert.','').replace('bertish.','')
                    roberta_params[key] = value
            self.bertish=RobertaModel.from_pretrained(pretrained_model_name_or_path,state_dict=roberta_params)
            self.bertish.train()
            self.text_projection = nn.Linear(1024, 768, bias=False)
            self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
            if not hasattr(user_defined_parameters,'mode'):
                user_defined_parameters['mode']='finetune'
            if user_defined_parameters['mode']=='finetune':
                self.clip = CLIPModel.from_pretrained(pretrained_model_name_or_path_vision) 
                for param in self.clip.parameters():
                    param.requires_grad = False

    def forward(self, inputs,feat=None):
        if self.bertish.device!=self.text_projection.weight.device:
            self.bertish=self.bertish.to(self.text_projection.weight.device)
        image_embeds=None
        text_embeds=None
        if 'input_ids' in inputs:
            text_outputs = self.bertish(input_ids=inputs['input_ids'].to(self.text_projection.weight.device),
            token_type_ids=inputs['token_type_ids'].to(self.text_projection.weight.device),
            attention_mask=inputs['attention_mask'].to(self.text_projection.weight.device))
            text_embeds = text_outputs[1]
            text_embeds = self.text_projection(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        if 'pixel_values' in inputs:
            vision_outputs = self.clip.vision_model(
                pixel_values=inputs['pixel_values'].to(self.text_projection.weight.device)
            )
            image_embeds = vision_outputs[1].detach()
            image_embeds = self.clip.visual_projection(image_embeds)
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


