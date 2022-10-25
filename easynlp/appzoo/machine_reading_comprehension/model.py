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
from ast import literal_eval

import torch
import torch.nn as nn

from ...utils import losses
from ..application import Application
from ...modelzoo import AutoConfig, AutoModel, BertModel


class MachineReadingComprehension(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()

        if kwargs.get('from_config'):
            # for evaluation and prediction
            print("============ from config ===============")
            self.config = kwargs.get('from_config')
            self._model = AutoModel.from_config(self.config)
        elif kwargs.get('user_defined_parameters') is not None and \
                "model_parameters" in kwargs.get('user_defined_parameters'):
            # for model random initialization
            print("============ from random initialization ===============")
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            user_defined_parameters = kwargs.get('user_defined_parameters')
            user_defined_parameters_dict = literal_eval(user_defined_parameters)
            self.config.update(user_defined_parameters_dict['model_parameters'])
            self._model = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            print("============ from pretrained model ===============")
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self._model = AutoModel.from_pretrained(pretrained_model_name_or_path)

        self.classifier = nn.Linear(self.config.hidden_size, 2)    # num_labels = 2 (start_logits & end_logits)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.init_weights()

    def init_weights(self):

        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, inputs):

        sequence_outputs = self._model(input_ids=inputs["input_ids"],
                                       attention_mask=inputs["attention_mask"],
                                       token_type_ids=inputs["token_type_ids"]
                                       )[0]                        # [batch_size, src_len, hidden_size]

        sequence_outputs = self.dropout(sequence_outputs)
        logits = self.classifier(sequence_outputs)                 # [batch_size, src_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)         # [batch_size, src_len, 1] [batch_size, src_len, 1]
        start_logits = start_logits.squeeze(-1)                    # [batch_size, src_len]
        end_logits = end_logits.squeeze(-1)                        # [batch_size, src_len]
        return {
            # "inputs": inputs,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "predictions": torch.argmax(logits, dim=1)
        }

    def compute_loss(self, forward_outputs, label_ids):
        """
        Args:
            forward_outputs (`dict`): a dict of results produced by the model (.forward()), it is worth noting that,
                the true labels (start_positions, end_positions) is carried in forward_outputs, and for loss computing

            label_ids: label_ids is not the true label, only for fitting the interface in trainer.py (L588),
                and it is not used for the computing of the loss.

        Returns:
            loss (`dict`): a dict containing the computed loss
        """

        start_logits, end_logits = forward_outputs["start_logits"], forward_outputs["end_logits"]   # [batch_size, src_len]
        start_positions, end_positions = label_ids[:, 0], label_ids[:, 1]                           # [batch_size]
        ignored_index = start_logits.size(1)                                                        # int (=src_len)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)                                        # float
        end_loss = loss_fct(end_logits, end_positions)                                              # float
        loss = (start_loss + end_loss) / 2                                                          # float

        return {
            "loss": loss
        }
