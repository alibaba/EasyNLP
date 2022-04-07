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
import json

import torch
import torch.nn as nn

from ...utils import losses
from ..application import Application
from ...modelzoo import AutoModel, AutoConfig


class SequenceLabeling(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()

        if kwargs.get('from_config'):
            # for evaluation and prediction
            self.config = kwargs.get('from_config')
            self.backbone = AutoModel.from_config(self.config)
        elif kwargs.get('user_defined_parameters') is not None and "model_parameters" in kwargs.get(
                'user_defined_parameters'):
            # for model random initialization
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            user_defined_parameters_dict = kwargs.get('user_defined_parameters')
            self.config.update(user_defined_parameters_dict['model_parameters'])
            self.backbone = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)

        if 'num_labels' in kwargs:
            self.config.num_labels = kwargs['num_labels']

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, inputs):
        if 'tok_to_orig_index' in inputs:
            inputs.pop('tok_to_orig_index')
        outputs = self.backbone(**inputs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return {
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1),
            "probabilities": torch.softmax(logits, dim=-1)
        }

    def compute_loss(self, forward_outputs, label_ids):
        logits = forward_outputs["logits"]
        return {
            "loss":
                losses.cross_entropy(logits.view(-1, self.config.num_labels), label_ids.view(-1))
        }
