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

from ...distillation.distill_application import DistillatoryBaseApplication
from ...fewshot_learning.fewshot_application import FewshotClassification, CPTClassification
from ...modelzoo import AutoConfig, AutoModel
from ...utils import losses
from ..application import Application


class TextMatch(Application):

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
            user_defined_parameters = kwargs.get('user_defined_parameters')
            user_defined_parameters_dict = json.loads(user_defined_parameters)
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
        outputs = self.backbone(**inputs)
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        return {
            "hidden": hidden_states,
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1),
            "probabilities": torch.softmax(logits, dim=-1)
        }

    def compute_loss(self, forward_outputs, label_ids):
        logits = forward_outputs["logits"]
        return {"loss": losses.cross_entropy(logits, label_ids)}


class TextMatchTwoTowerV1(Application):

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
            user_defined_parameters = kwargs.get('user_defined_parameters')
            user_defined_parameters_dict = json.loads(user_defined_parameters)
            self.config.update(user_defined_parameters_dict)
            self.backbone = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)
        if 'num_labels' in kwargs:
            self.config.num_labels = kwargs['num_labels']

        if kwargs['user_defined_parameters'] is not None and "app_parameters" in kwargs[
                'user_defined_parameters']:
            user_defined_parameters_dict = kwargs['user_defined_parameters']
            self.config.loss_type = user_defined_parameters_dict["app_parameters"]['loss_type']
            self.config.margin = user_defined_parameters_dict["app_parameters"]['margin']
            self.config.gamma = user_defined_parameters_dict["app_parameters"]['gamma']

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.init_weights()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, inputs):
        inputs_a = {}
        inputs_a['input_ids'] = inputs.pop('input_ids_a')
        inputs_a['token_type_ids'] = inputs.pop('token_type_ids_a')
        inputs_a['attention_mask'] = inputs.pop('attention_mask_a')
        inputs_b = {}
        inputs_b['input_ids'] = inputs.pop('input_ids_b')
        inputs_b['token_type_ids'] = inputs.pop('token_type_ids_b')
        inputs_b['attention_mask'] = inputs.pop('attention_mask_b')

        outputs_a = self.backbone(**inputs_a)
        outputs_b = self.backbone(**inputs_b)

        pooler_output_a = outputs_a.pooler_output
        pooler_output_a = self.dropout(pooler_output_a)

        pooler_output_b = outputs_b.pooler_output
        pooler_output_b = self.dropout(pooler_output_b)
        logits = self.cos(pooler_output_a, pooler_output_b)
        return {
            "logits": logits,
            "pooler_output_a": pooler_output_a,
            "pooler_output_b": pooler_output_b
        }

    def compute_loss(self, forward_outputs, label_ids):
        emb1 = forward_outputs["pooler_output_a"]
        emb2 = forward_outputs["pooler_output_b"]

        assert self.config.loss_type in ["hinge_loss", "circle_loss"]
        if self.config.margin == -1.0:
            margin = 0.3 if self.loss_type == "hinge_loss" else 0.45
        else:
            margin = self.config.margin

        if self.config.loss_type == "hinge_loss":
            return {"loss": losses.matching_embedding_hinge_loss(emb1, emb2, margin=margin)}
        elif self.config.loss_type == "circle_loss":
            return {
                "loss":
                    losses.matching_embedding_circle_loss(emb1,
                                                          emb2,
                                                          margin=margin,
                                                          gamma=self.config.gamma)
            }


class TextMatchTwoTower(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super(TextMatchTwoTower, self).__init__()
        if  kwargs.get('from_config'):
            # for evaluation and prediction
            self.config = kwargs.get('from_config')
            self.backbone = AutoModel.from_config(self.config)

        elif kwargs.get('user_defined_parameters') is not None and "model_parameters" in kwargs.get(
                'user_defined_parameters'):
            # for model random initialization
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            user_defined_parameters_dict = kwargs.get('user_defined_parameters')
            self.config.update(user_defined_parameters_dict)
            self.backbone = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)

        if 'num_labels' in kwargs:
            self.config.num_labels = kwargs['num_labels']

        if kwargs.get('user_defined_parameters') is not None and "app_parameters" in kwargs['user_defined_parameters']:
            user_defined_parameters_dict = kwargs['user_defined_parameters']
            self.config.loss_type = user_defined_parameters_dict["app_parameters"]['loss_type']
            self.config.margin = user_defined_parameters_dict["app_parameters"]['margin']
            self.config.gamma = user_defined_parameters_dict["app_parameters"]['gamma']
            self.config.embedding_size = user_defined_parameters_dict["app_parameters"]['embedding_size']

        self.emb_layer = nn.Linear(self.config.hidden_size, self.config.embedding_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.init_weights()

    def init_weights(self):
        self.emb_layer.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.emb_layer.bias.data.zero_()

    def forward(self, inputs):
        if 'input_ids' in inputs:
            # for prediction
            outputs = self.backbone(**inputs)
            pooler_output = outputs.pooler_output
            emb_output = self.emb_layer(pooler_output)
            return {
                "emb_output": emb_output
            }
        else:
            #for training and evaluation
            inputs_a = {}
            inputs_a['input_ids'] = inputs.pop('input_ids_a')
            inputs_a['token_type_ids'] = inputs.pop('token_type_ids_a')
            inputs_a['attention_mask'] = inputs.pop('attention_mask_a')
            inputs_b = {}
            inputs_b['input_ids'] = inputs.pop('input_ids_b')
            inputs_b['token_type_ids'] = inputs.pop('token_type_ids_b')
            inputs_b['attention_mask'] = inputs.pop('attention_mask_b')

            outputs_a = self.backbone(**inputs_a)
            outputs_b = self.backbone(**inputs_b)

            pooler_output_a = outputs_a.pooler_output
            pooler_output_a = self.dropout(pooler_output_a)
            emb_output_a = self.emb_layer(pooler_output_a)

            pooler_output_b = outputs_b.pooler_output
            pooler_output_b = self.dropout(pooler_output_b)
            emb_output_b = self.emb_layer(pooler_output_b)
            # TODO: for evaluateion?
            logits = self.cos(emb_output_a, emb_output_b)
            return {
                "logits": logits,
                "emb_output_a": emb_output_a,
                "emb_output_b": emb_output_b,
                "pooler_output_a": pooler_output_a,
                "pooler_output_b": pooler_output_b
            }

    def compute_loss(self, forward_outputs, label_ids):
        emb1 = forward_outputs["emb_output_a"]
        emb2 = forward_outputs["emb_output_b"]

        assert self.config.loss_type in ["hinge_loss", "circle_loss"]
        if self.config.margin == -1.0:
            margin = 0.3 if self.loss_type == "hinge_loss" else 0.45
        else:
            margin = self.config.margin

        if self.config.loss_type == "hinge_loss":
            return {
                "loss": losses.matching_embedding_hinge_loss(emb1, emb2, margin=margin)
            }
        elif self.config.loss_type == "circle_loss":
            return {
                "loss": losses.matching_embedding_circle_loss(emb1, emb2, margin=margin, gamma=self.config.gamma)
            }


class DistillatoryTextMatch(DistillatoryBaseApplication, TextMatch):
    pass

class FewshotSingleTowerTextMatch(FewshotClassification):
    pass

class CptFewshotSingleTowerTextMatch(CPTClassification):
    pass
