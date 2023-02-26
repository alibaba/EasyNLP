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
from torch import nn

from ..appzoo.application import Application
from ..modelzoo import AutoConfig, AutoModel
from ..utils.losses import cross_entropy, mse_loss, soft_cross_entropy


class MetaTeacherForSequenceClassification(Application):
    r"""

    An application class for supporting meta-teacher learning.
    args:
        pretrained_model_name_or_path: the path of model.

        num_labels: the number of labels.

        num_domains: the number of domains.

    Example:

    ```python
    >>> from easynlp.distillation.meta_modeling import MetaTeacherForSequenceClassification
    >>> path = "bert-base-uncased" # using huggingface model
    >>> model = MetaTeacherForSequenceClassification(pretrained_model_name_or_path=path, num_labels=2, num_domains=4)

    >>> path = "checkpoint-path" # using self-defined model base on Application
    >>> model = MetaTeacherForSequenceClassification.from_pretrained(path)
    ```
    """
    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()
        if kwargs.get('from_config'):
            # for evaluation and prediction
            self.config = kwargs.get('from_config')
            self.backbone = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            self.config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(
                pretrained_model_name_or_path)
        if 'num_labels' in kwargs:
            self.config.num_labels = kwargs['num_labels']
        if 'num_domains' in kwargs:
            self.config.num_domains = kwargs['num_domains']
        self.classifier = nn.Linear(self.config.hidden_size,
                                    self.config.num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.domain_embedding = nn.Embedding(self.config.num_domains,
                                             self.config.hidden_size)
        self.corrupt_dense = nn.Linear(self.config.hidden_size,
                                       self.config.hidden_size)
        self.domain_classifier = nn.Linear(self.config.hidden_size,
                                           self.config.num_domains)

    def forward(self, inputs):
        r"""
            input_ids, attention_mask, token_type_ids is same to class PreTrainedModel
            domain_ids: the domain id of data. Shape: [, batchsize]

        """
        backbone_output = self.backbone(inputs['input_ids'],
                                        inputs['attention_mask'],
                                        inputs['token_type_ids'],
                                        output_hidden_states=True,
                                        output_attentions=True)

        logits = self.classifier(torch.relu(backbone_output.pooler_output))
        # Domain corrupt
        domain_embedded = self.domain_embedding(inputs['domain_ids'])
        content_tensor = torch.mean(backbone_output.last_hidden_state[:,
                                                                      1:, :],
                                    dim=1)
        content_output = torch.tanh(
            self.corrupt_dense(domain_embedded + content_tensor))
        content_output = self.dropout(content_output)
        domain_logits = self.domain_classifier(content_output)

        return {
            'logits': logits,
            'domain_logits': domain_logits,
            'attentions': backbone_output.attentions,
            'hidden': backbone_output.hidden_states,
            'predictions': torch.argmax(logits, dim=-1),
            'probabilities': torch.softmax(logits, dim=-1)
        }

    def compute_loss(self,
                     forward_outputs,
                     label_ids,
                     use_domain_loss=True,
                     use_sample_weights=True,
                     **kwargs):
        logits = forward_outputs['logits']
        per_instance_loss = cross_entropy(logits, label_ids, reduction='none')
        if use_domain_loss:
            shuffled_domain_ids = kwargs['domain_ids'][torch.randperm(
                kwargs['domain_ids'].shape[0])]
            domain_lossed = cross_entropy(forward_outputs['domain_logits'],
                                          shuffled_domain_ids,
                                          reduction='none')
            per_instance_loss += kwargs['domain_loss_weight'] * domain_lossed

        if use_sample_weights:
            loss = torch.mean(per_instance_loss * kwargs['sample_weights'])
        else:
            loss = torch.mean(per_instance_loss)
        return {'loss': loss}


class MetaStudentForSequenceClassification(Application):
    r"""

    An application class for supporting meta-distillation.
    args is same to MetaTeacherForSequenceClassification

    You can use the checkpoint from MetaTeacherForSequenceClassification to initialize this model.
    Example:

    ```python
    >>> path = "checkpoint-path-from-MetaTeacherForSequenceClassification"
    >>> model = MetaTeacherForSequenceClassification.from_pretrained(path)
    ```
    """
    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()
        if kwargs.get('from_config'):
            # for evaluation and prediction
            self.config = kwargs.get('from_config')
            self.backbone = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            self.config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(
                pretrained_model_name_or_path)
        if 'num_labels' in kwargs:
            self.config.num_labels = kwargs['num_labels']
        if 'num_domains' in kwargs:
            self.config.num_domains = kwargs['num_domains']

        self.config.fit_size = kwargs[
            'fit_size'] if 'fit_size' in kwargs else 768
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size,
                                    self.config.num_labels)
        self.domain_embedding = nn.Embedding(self.config.num_domains,
                                             self.config.hidden_size)
        self.corrupt_dense = nn.Linear(self.config.hidden_size,
                                       self.config.hidden_size)
        self.domain_classifier = nn.Linear(self.config.hidden_size,
                                           self.config.num_domains)
        self.fit_dense = nn.Linear(self.config.hidden_size,
                                   self.config.fit_size)

    def forward(self, inputs, is_student=False, distill_stage='all'):
        """Pre-trained distillation when distill_stage is "first", return [attentions, sequence_output, domain_content_output].

        Downstream task distillation when distill_stage is "second", return [logits].
        This approach is to solve the problem that distributed training cannot find the tensor.
        """

        if distill_stage not in ['first', 'second', 'all']:
            raise RuntimeError(
                'The distill_stage flag must be one of [first, second]')
        if distill_stage == 'first' or distill_stage == 'all':
            backbone_output = self.backbone(inputs['input_ids'],
                                            inputs['attention_mask'],
                                            inputs['token_type_ids'],
                                            output_hidden_states=True,
                                            output_attentions=True)

            domain_embedded = self.domain_embedding(
                inputs['domain_ids']).squeeze()
            content_tensor = torch.mean(
                backbone_output.last_hidden_state[:, 1:, :], dim=1)
            domain_content_output = self.fit_dense(
                self.corrupt_dense(domain_embedded + content_tensor))

            if is_student:
                sequence_output = [
                    self.fit_dense(hidden_state)
                    for hidden_state in backbone_output.hidden_states
                ]
            else:
                sequence_output = backbone_output.hidden_states

            if distill_stage == 'first':
                return {
                    'attentions': backbone_output.attentions,
                    'sequence_output': sequence_output,
                    'domain_content_output': domain_content_output
                }
            else:
                logits = self.classifier(
                    torch.relu(backbone_output.pooler_output))
                return {
                    'logits': logits,
                    'attentions': backbone_output.attentions,
                    'sequence_output': sequence_output,
                    'domain_content_output': domain_content_output
                }

        else:
            backbone_output = self.backbone(inputs['input_ids'],
                                            inputs['attention_mask'],
                                            inputs['token_type_ids'],
                                            output_hidden_states=True,
                                            output_attentions=True)
            logits = self.classifier(torch.relu(backbone_output.pooler_output))
            return {'logits': logits}

    def compute_loss(self, **kwargs):
        """When distill_stage is first:

        The student model will fit the teacher model of [attention, representation, domain].
        When distill_stage is second:
            The student model will use distillation loss to fit the logits of the teacher model.
        """
        distill_stage = kwargs['distill_stage']

        local_rank = kwargs['local_rank']
        # Prepare parameters
        if distill_stage == 'first':
            student_atts = kwargs['student_atts']
            student_reps = kwargs['student_reps']
            student_domain_rep = kwargs['student_domain_rep']

            teacher_atts = kwargs['teacher_atts']
            teacher_reps = kwargs['teacher_reps']
            teacher_domain_rep = kwargs['teacher_domain_rep']
            grt_sample_weights = kwargs['grt_sample_weights']

            domain_loss_weight = kwargs['domain_loss_weight']
            sample_weights = kwargs['sample_weights']
            final_sample_weights = (1 + sample_weights) * grt_sample_weights

        else:
            teacher_logits = kwargs['teacher_logits']
            student_logits = kwargs['student_logits']
            T = kwargs['T']

        if distill_stage == 'first':
            att_loss = 0.
            rep_loss = 0.
            domain_loss = 0.

            teacher_layer_num = len(teacher_atts)
            student_layer_num = len(student_atts)
            assert teacher_layer_num % student_layer_num == 0 and teacher_layer_num >= student_layer_num
            layers_per_block = int(teacher_layer_num / student_layer_num)
            new_teacher_atts = [
                teacher_atts[i * layers_per_block + layers_per_block - 1]
                for i in range(student_layer_num)
            ]

            # attention loss
            for student_att, teacher_att in zip(student_atts,
                                                new_teacher_atts):
                # Deleting small att
                student_att = torch.where(
                    student_att <= -1e2,
                    torch.zeros_like(student_att).to(local_rank), student_att)
                teacher_att = torch.where(
                    teacher_att <= -1e2,
                    torch.zeros_like(teacher_att).to(local_rank), teacher_att)
                tmp_loss = mse_loss(student_att, teacher_att)
                att_loss += tmp_loss

            # representations loss
            new_teacher_reps = [
                teacher_reps[i * layers_per_block]
                for i in range(student_layer_num + 1)
            ]
            new_student_reps = student_reps
            for student_rep, teacher_rep in zip(new_student_reps,
                                                new_teacher_reps):
                tmp_loss = mse_loss(student_rep, teacher_rep)
                rep_loss += tmp_loss

            # domain loss
            domain_loss += mse_loss(teacher_domain_rep, student_domain_rep)

            # final_loss
            loss = rep_loss.mean(-1).mean(-1) * final_sample_weights + \
                   att_loss.mean(-1).mean(-1).mean(-1) * final_sample_weights + \
                   domain_loss_weight * domain_loss.mean(-1).mean(-1) * final_sample_weights

            loss = loss.mean()
            return {
                'loss': loss,
                'att_loss': att_loss,
                'rep_loss': rep_loss,
                'domain_loss': domain_loss
            }

        else:
            cls_loss = 0.
            cls_loss = soft_cross_entropy(student_logits / T,
                                          teacher_logits / T)
            loss = cls_loss
            return {'loss': loss}
