# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and Princeton Natural Language Processing.
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

"""Custom models for few-shot learning specific operations."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaModel,
)

from torch.nn import MSELoss

logger = logging.getLogger(__name__)


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, "bert"):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(
        new_num_types, old_token_type_embeddings.weight.size(1)
    )
    if not random_segment:
        new_token_type_embeddings.weight.data[
            : old_token_type_embeddings.weight.size(0)
        ] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, "bert"):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls_head = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def _compute_mlm_loss(self, logits, labels, weights=None):
        if self.num_labels == 1:
            # Regression task
            loss_fct = nn.KLDivLoss(log_target=True)
            labels = torch.stack(
                [
                    1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    (labels.view(-1) - self.lb) / (self.ub - self.lb),
                ],
                -1,
            )
            loss = loss_fct(logits.view(-1, 2), labels)
        else:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            if weights is not None:
                loss = torch.mean(loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) * weights)
            else:
                loss = torch.mean(loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)))

        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return (
                    torch.zeros(1, out=prediction_mask_scores.new()),
                    prediction_mask_scores,
                )
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = self._compute_mlm_loss(logits, labels)

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


class BertCRSDistillTeacher(BertForPromptFinetuning):
    def __init__(self, config):
        super().__init__(config)
        # The number of student layer for intermidate kd.
        self.target_num = 4
        self.init_weights()

    @staticmethod
    def _compute_cls_logits(mask_scores, label_word_list):
        logits = [
            torch.unsqueeze(mask_scores[:, label_id], dim=-1)
            for label_id in label_word_list
        ]
        return torch.cat(logits, -1)

    @staticmethod
    def _compute_mse_loss(high_hidden, low_hidden):
        loss_mse = MSELoss()
        loss = loss_mse(high_hidden, low_hidden)
        return loss

    def forward(self, input_ids=None, attention_mask=None, mask_pos=None, labels=None):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get <mask> token representation
        intermediate_outputs = [
            hid_out.detach() for hid_out in outputs.hidden_states[1:-1]
        ]
        intermediate_mask_outputs = [
            hid_out[torch.arange(batch_size), mask_pos]
            for hid_out in intermediate_outputs
        ]

        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        intermediate_mask_outputs.append(sequence_mask_output)

        self_loss = []
        for i, inter_state in enumerate(intermediate_mask_outputs, 0):
            current_block = i // self.target_num + 1
            if i %  self.target_num == 0:
                self_loss.append(self._compute_mse_loss(inter_state, 
                    intermediate_mask_outputs[current_block * self.target_num - 1]))

        # For exporting internal layer logits
        if self.eval:        
            out_inter_logits = [per_layer_logits for i, per_layer_logits in 
                enumerate(intermediate_mask_outputs, 1)]
            inter_logits = torch.cat(
                [_logits.unsqueeze(-1) for _logits in out_inter_logits], dim=-1
            )
        else:
            inter_logits = None

        p_loss = sum(self_loss)

        return p_loss, inter_logits

class BertDistillStudent(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.alpha = config.alpha
        self.T = config.temperature
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, self.num_labels)
        self.bert_tiny = config.bert_tiny
        self.init_weights()
    
    def _compute_kd_loss(self, logits, teacher_cls_logits, T, weights=None):
        """Dark Knowledge Distillation Loss"""
        return soft_cross_entropy(logits / T, teacher_cls_logits / T, weights)
    
    def _compute_cls_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    def _soft_cross_entropy(self, predicts, targets):
        student_likelihood = nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        teacher_cls_logits=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        logits = self.cls(torch.relu(outputs.pooler_output))
        if not self.training:
            loss = self._compute_cls_loss(logits, labels)
            return (loss, logits)
        
        if self.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)
        
        if self.bert_tiny:
            assert teacher_cls_logits is not None
            loss = self._soft_cross_entropy(logits / self.T, teacher_cls_logits / self.T)
            return (loss, logits)
        
        cls_loss = self._compute_cls_loss(logits, labels)
        kd_loss = self._compute_kd_loss(logits, teacher_cls_logits, self.T)
        loss = self.alpha * kd_loss + (1 - self.alpha) * cls_loss
        return (loss, logits)

class BertCRSDistillStudent(BertForPromptFinetuning):
    def __init__(self, config):
        super().__init__(config)

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.T = config.temperature
        self.init_weights()

    def _compute_kd_loss(self, logits, teacher_cls_logits, T, weights=None):
        """Dark Knowledge Distillation Loss"""
        return soft_cross_entropy(logits / T, teacher_cls_logits / T, weights)


    def _compute_ild_loss(self, pseudo_logits, teacher_inter_logits, neighbor, T):
        """Intermediate Layers Distillation Loss"""
        loss_mse = MSELoss()
        # Choose one
        # teacher_inter_logits = teacher_inter_logits[..., [14, 17, 20, 23]]
        teacher_inter_logits = teacher_inter_logits[..., [20, 21, 22, 23]]
        return loss_mse(pseudo_logits, teacher_inter_logits)

    def _compute_loss(
        self, labels, logits, pseudo_logits, teacher_cls_logits, teacher_inter_logits, weights, high_acc_prob=None
    ):
        mlm_weight = 1 - self.alpha - self.beta - self.gamma
        kd_weight = self.alpha
        ild_weight = self.beta
        kd_loss_high_weight = self.gamma
        mlm_loss = self._compute_mlm_loss(logits, labels, weights) if mlm_weight > 0 else 0

        kd_loss = (
            self._compute_kd_loss(logits, teacher_cls_logits, self.T, weights)
            if kd_weight > 0
            else 0
        )
        
        ild_loss = (
            self._compute_ild_loss(pseudo_logits, teacher_inter_logits, 0, self.T)
            if ild_weight > 0
            else 0
        ) if teacher_inter_logits is not None else 0

        # soft label loss
        kd_loss_high = (
            self._compute_kd_loss(logits, high_acc_prob, self.T, weights)
            if high_acc_prob is not None
            else 0
        )   

        return (
            mlm_weight * mlm_loss
            + kd_weight * kd_loss
            + self.beta * ild_loss
            + kd_loss_high_weight * kd_loss_high
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        teacher_cls_logits=None,
        teacher_inter_logits=None,
        weights=None,
        high_acc_prob=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        if teacher_cls_logits is None:
            print()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        # Get <mask> token representation
        sequence_output = outputs.last_hidden_state
        intermediate_outputs = outputs.hidden_states[1:-1]

        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        intermediate_mask_outputs = [
            hid_out[torch.arange(batch_size), mask_pos]
            for hid_out in intermediate_outputs
        ]

        def _compute_cls_logits(_mask_scores):
            _logits = []
            for label_id in range(len(self.label_word_list)):
                _logits.append(
                    _mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
                )
            return torch.cat(_logits, -1)

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls_head(sequence_mask_output)
        logits = _compute_cls_logits(prediction_mask_scores)

        if not self.training:
            loss = self._compute_mlm_loss(logits, labels, weights)
            return (loss, logits)

        intermediate_mask_outputs.append(sequence_mask_output)
        pseudo_logits = torch.cat(
            [_logits.unsqueeze(-1) for _logits in intermediate_mask_outputs], dim=-1
        )
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = self._compute_loss(
            labels, logits, pseudo_logits, teacher_cls_logits, teacher_inter_logits, weights, high_acc_prob
        )

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )
        return ((loss,) + output) if loss is not None else output


class RobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        # self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def _compute_mlm_loss(self, logits, labels):
        if self.num_labels == 1:
            # Regression task
            loss_fct = nn.KLDivLoss(log_target=True)
            labels = torch.stack(
                [
                    1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                    (labels.view(-1) - self.lb) / (self.ub - self.lb),
                ],
                -1,
            )
            loss = loss_fct(logits.view(-1, 2), labels)
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(input_ids, attention_mask=attention_mask)

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return (
                    torch.zeros(1, out=prediction_mask_scores.new()),
                    prediction_mask_scores,
                )
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(
                prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
            )
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = None
        if labels is not None:
            loss = self._compute_mlm_loss(logits, labels)

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )

        return ((loss,) + output) if loss is not None else output


class RobertaCRSDistillTeacher(RobertaForPromptFinetuning):
    def __init__(self, config):
        super().__init__(config)
        # The number of student layer for intermidate kd.
        self.target_num = 4
        self.fit_dense = torch.nn.Linear(config.hidden_size, 512)

        self.init_weights()

    @staticmethod
    def _compute_cls_logits(mask_scores, label_word_list):
        logits = [
            torch.unsqueeze(mask_scores[:, label_id], dim=-1)
            for label_id in label_word_list
        ]
        return torch.cat(logits, -1)

    @staticmethod
    def _compute_mse_loss(high_hidden, low_hidden):
        loss_mse = MSELoss()
        loss = loss_mse(high_hidden, low_hidden)
        return loss

    def forward(self, input_ids=None, attention_mask=None, mask_pos=None, labels=None):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get <mask> token representation
        intermediate_outputs = [
            hid_out.detach() for hid_out in outputs.hidden_states[1:-1]
        ]
        intermediate_mask_outputs = [
            hid_out[torch.arange(batch_size), mask_pos]
            for hid_out in intermediate_outputs
        ]

        sequence_output = outputs[0]
        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]

        intermediate_mask_outputs.append(sequence_mask_output)

        self_loss = []
        for i, inter_state in enumerate(intermediate_mask_outputs, 0):
            current_block = i // self.target_num + 1
            if i %  self.target_num == 0:
                self_loss.append(self._compute_mse_loss(inter_state, 
                    intermediate_mask_outputs[current_block * self.target_num - 1]))

        # For exporting internal layer logits
        if self.eval:        
            out_inter_logits = [self.fit_dense(per_layer_logits) for i, per_layer_logits in 
                enumerate(intermediate_mask_outputs, 1)]
            inter_logits = torch.cat(
                [_logits.unsqueeze(-1) for _logits in out_inter_logits], dim=-1
            )
        else:
            inter_logits = None

        p_loss = sum(self_loss)

        return p_loss, inter_logits


class RobertaCRSDistillStudent(RobertaForPromptFinetuning):
    def __init__(self, config):
        super().__init__(config)

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.T = config.temperature
        self.init_weights()

    def _compute_kd_loss(self, logits, teacher_cls_logits, T, weights=None):
        """Dark Knowledge Distillation Loss"""
        return soft_cross_entropy(logits / T, teacher_cls_logits / T, weights)

    def _compute_ild_loss(self, pseudo_logits, teacher_inter_logits, T):
        """Intermediate Layers Distillation Loss"""
        loss_mse = MSELoss()
        # Choose one
        # teacher_inter_logits = teacher_inter_logits[..., [14, 17, 20, 23]]
        teacher_inter_logits = teacher_inter_logits[..., [20, 21, 22, 23]]
        return loss_mse(pseudo_logits, teacher_inter_logits)

    def _compute_loss(
        self, labels, logits, pseudo_logits, teacher_cls_logits, teacher_inter_logits, weights, high_acc_prob=None
    ):
        mlm_weight = 1 - self.alpha - self.beta - self.gamma
        kd_weight = self.alpha
        ild_weight = self.beta
        kd_loss_high_weight = self.gamma
        mlm_loss = self._compute_mlm_loss(logits, labels, weights) if mlm_weight > 0 else 0

        kd_loss = (
            self._compute_kd_loss(logits, teacher_cls_logits, self.T, weights)
            if kd_weight > 0
            else 0
        )
        
        ild_loss = (
            self._compute_ild_loss(pseudo_logits, teacher_inter_logits, 0, self.T)
            if ild_weight > 0
            else 0
        ) if teacher_inter_logits is not None else 0

        # soft label loss
        kd_loss_high = (
            self._compute_kd_loss(logits, high_acc_prob, self.T, weights)
            if high_acc_prob is not None
            else 0
        )   

        return (
            mlm_weight * mlm_loss
            + kd_weight * kd_loss
            + self.beta * ild_loss
            + kd_loss_high_weight * kd_loss_high
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        teacher_cls_logits=None,
        teacher_inter_logits=None,
        weights=None,
        high_acc_prob=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get <mask> token representation
        sequence_output = outputs.last_hidden_state
        intermediate_outputs = outputs.hidden_states[1:-1]

        sequence_mask_output = sequence_output[
            torch.arange(sequence_output.size(0)), mask_pos
        ]
        intermediate_mask_outputs = [
            hid_out[torch.arange(batch_size), mask_pos]
            for hid_out in intermediate_outputs
        ]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)
        pseudo_mask_scores = [
            p_head(hid_out)
            for p_head, hid_out in zip(self.pseudo_heads, intermediate_mask_outputs)
        ]

        # Return logits for each label
        def _compute_cls_logits(_mask_scores):
            _logits = []
            for label_id in range(len(self.label_word_list)):
                _logits.append(
                    _mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)
                )
            return torch.cat(_logits, -1)

        logits = _compute_cls_logits(prediction_mask_scores)
        pseudo_logits_list = list(map(_compute_cls_logits, pseudo_mask_scores))
        pseudo_logits_list.append(logits)
        pseudo_logits = torch.cat(
            [_logits.unsqueeze(-1) for _logits in pseudo_logits_list], dim=-1
        )

        assert pseudo_logits.size(-1) == self.config.num_hidden_layers

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        loss = self._compute_loss(
            labels, logits, pseudo_logits, teacher_cls_logits, teacher_inter_logits
        )

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,
            )

        return ((loss,) + output) if loss is not None else output

def soft_cross_entropy(predicts, targets, weights=None):
    student_likelihood = F.log_softmax(predicts, dim=-1)
    targets_prob = F.softmax(targets, dim=-1)
    if weights == None:
        return (- targets_prob * student_likelihood).mean()
    else:
        batch_weights_loss = (- targets_prob * student_likelihood).mean(-1) * weights
        return batch_weights_loss.mean()