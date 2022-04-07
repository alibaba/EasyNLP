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

from ...utils.losses import cross_entropy
from ..application import Application
from ...modelzoo import AutoConfig, AutoModelForMaskedLM


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
    def forward(self, inputs):
        if 'mask_span_indices' in inputs:
            inputs.pop('mask_span_indices')
        outputs = self.backbone(**inputs)
        return {"logits": outputs.logits}

    def compute_loss(self, forward_outputs, label_ids, insert_know_labels=None):
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
        return {"loss": total_loss}
