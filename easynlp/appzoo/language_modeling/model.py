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
from typing import Optional
import torch
import torch.nn as nn
from ...utils.losses import cross_entropy
from ..application import Application
from ...modelzoo import AutoConfig, AutoModelForMaskedLM

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class LanguageModeling(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()
        if kwargs.get('from_config'):
            self.config = kwargs.get('from_config')
            self.backbone = AutoModelForMaskedLM.from_config(self.config)
        # for pretrained model, initialize from the pretrained model
        else:
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        temp = 0.5
        self.CosSim = Similarity(temp=temp)
        
    def forward(self, inputs):
        if 'mask_span_indices' in inputs:
            inputs.pop('mask_span_indices')
        outputs = self.backbone(**inputs)
        return {"logits": outputs.logits, "hidden_states":outputs.hidden_states}
    
    def compute_simcse(self, original_outputs: torch.Tensor, forward_outputs: torch.Tensor):
        original_hidden_states = original_outputs['hidden_states'].unsqueeze(-2)
        loss = nn.CrossEntropyLoss()
        
        forward_outputs = torch.mean(forward_outputs, dim=-2)
        cos_result = self.CosSim(original_hidden_states, forward_outputs)
        cos_result_size = cos_result.size()
        cos_result = cos_result.view(-1, cos_result_size[-1])
        labels = torch.zeros(cos_result.size(0), device=original_outputs['hidden_states'].device).long()
        
        loss_ = loss(cos_result, labels)
        return loss_
    
    def compute_loss(self, forward_outputs, label_ids, insert_know_labels=None,
                     constrast_learning_flag: bool = False, 
                     positive_negative_results: Optional[torch.Tensor] = None):
        prediction_scores = forward_outputs["logits"]
        total_loss = None
        if insert_know_labels == None:
            total_loss = cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                        label_ids.view(-1))
        else:
            masked_lm_loss = cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                        label_ids.view(-1))
            decode_know_loss = cross_entropy(prediction_scores.view(-1, self.config.vocab_size),
                                        insert_know_labels.view(-1))
            total_loss = masked_lm_loss + decode_know_loss
           
        cl_loss = 0.0    
        if constrast_learning_flag:
            coeff = 0.09
            cl_loss = self.compute_simcse(forward_outputs, positive_negative_results) * coeff
            
        return {"loss": total_loss, 'cl_loss': cl_loss}
