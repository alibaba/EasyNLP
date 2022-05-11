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

import time

import torch

from .trainer import Trainer


class DistillatoryTrainer(Trainer):
    def __init__(self, user_defined_parameters, **kwargs):
        super(DistillatoryTrainer, self).__init__(**kwargs)

        if not isinstance(user_defined_parameters, dict):
            raise TypeError(
                '`user_defined_parameters` should be a characterized '
                'dictionary data structure.')
        self.distillation_params = user_defined_parameters.get(
            'app_parameters')

    def train(self):
        self.log_train_infos()
        args = self.args

        distillation_params = self.distillation_params

        for _epoch in range(self._start_epoch, int(args.epoch_num)):
            self.before_epoch(_epoch)

            for _step, batch in enumerate(self._train_loader):
                if self._global_step + 1 < self._start_global_step:
                    if (_step + 1) % args.gradient_accumulation_steps == 0:
                        self._global_step += 1
                    continue
                self.before_iter()

                batch = {
                    key: val.to(args.local_rank) if isinstance(
                        val, torch.Tensor) else val
                    for key, val in batch.items()
                }
                label_ids = batch.pop('label_ids')

                # type, temperature, alpha, teacher_logits...
                distillation_params['teacher_logits'] = batch.pop(
                    'teacher_logits')

                forward_outputs = self._model(batch)
                loss_dict = self.model_module.compute_loss(
                    forward_outputs,
                    label_ids,
                    **distillation_params,
                )

                _loss = loss_dict['loss']
                if args.n_gpu > 1:
                    _loss = _loss.mean()
                if args.gradient_accumulation_steps > 1:
                    _loss = _loss / args.gradient_accumulation_steps

                _loss.backward()

                self.after_iter(_step, _epoch, loss_dict)

            self.after_epoch()

        self.after_train()


class MetaTeacherTrainer(Trainer):
    def __init__(self, model, train_dataset, evaluator, **kwargs):
        super().__init__(model, train_dataset, evaluator, **kwargs)
        user_defined_parameters = kwargs['user_defined_parameters']
        self.use_domain_loss = True if user_defined_parameters[
            'use_domain_loss'] == 'True' else False
        self.use_sample_weights = True if user_defined_parameters[
            'use_sample_weights'] == 'True' else False
        self.domain_loss_weight = float(
            user_defined_parameters['domain_loss_weight'])

    def train(self):
        self.log_train_infos()
        args = self.args
        for _epoch in range(self._start_epoch, int(args.epoch_num)):
            self.before_epoch(_epoch)
            for _step, batch in enumerate(self._train_loader):
                if self._global_step + 1 < self._start_global_step:
                    if (_step + 1) % args.gradient_accumulation_steps == 0:
                        self._global_step += 1
                    continue
                self.before_iter()
                batch = {
                    key: val.to(args.local_rank) if isinstance(
                        val, torch.Tensor) else val
                    for key, val in batch.items()
                }
                label_ids = batch.pop('label_ids')
                forward_outputs = self._model(batch)
                loss_input = {
                    'forward_outputs': forward_outputs,
                    'label_ids': label_ids,
                    'use_domain_loss': self.use_domain_loss,
                    'use_sample_weights': self.use_sample_weights,
                    'domain_ids': batch['domain_ids'],
                    'sample_weights': batch['sample_weights'],
                    'domain_loss_weight': self.domain_loss_weight
                }
                loss_dict = self.model_module.compute_loss(**loss_input)
                _loss = loss_dict['loss']
                if args.n_gpu > 1:
                    _loss = _loss.mean()
                if args.gradient_accumulation_steps > 1:
                    _loss = _loss / args.gradient_accumulation_steps
                _loss.backward()

                self.after_iter(_step, _epoch, loss_dict)
            self.after_epoch()
        print('Training Time: {}, rank {}, gsteps {}'.format(
            time.time() - self._start_time, args.rank, self._global_step))
        self.after_train()


class MetaDistillationTrainer(Trainer):
    def __init__(self, student_model, teacher_model, train_dataset, evaluator,
                 **kwargs):
        super().__init__(student_model, train_dataset, evaluator, **kwargs)
        self._teacher = None
        self.set_teacher_model(teacher_model)
        user_defined_parameters = kwargs['user_defined_parameters']
        self.domain_loss_weight = float(
            user_defined_parameters['domain_loss_weight'])
        self.T = int(user_defined_parameters['T'])
        self.distill_stage = user_defined_parameters['distill_stage']
        self.num_labels = student_model.config.num_labels
        if self.distill_stage not in ['first', 'second']:
            raise RuntimeError(
                'The distill_stage flag must be one of [first, second]')

    def set_teacher_model(self, model):
        if self.args.use_torchacc:
            self._teacher = model.to(self._device)
        elif self.args.n_gpu == 1:
            self._teacher = model.to(self.args.local_rank)
        elif self.args.n_gpu > 1:
            self._teacher = torch.nn.parallel.DistributedDataParallel(
                model.to(self.args.local_rank),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True)
        else:
            raise Exception('CPU Training is not supported.')

    def train(self):
        self.log_train_infos()
        args = self.args
        for _epoch in range(self._start_epoch, int(args.epoch_num)):
            self.before_epoch(_epoch)
            for _step, batch in enumerate(self._train_loader):
                if self._global_step + 1 < self._start_global_step:
                    if (_step + 1) % args.gradient_accumulation_steps == 0:
                        self._global_step += 1
                    continue
                self.before_iter()
                input_dict = {
                    'input_ids': batch['input_ids'].to(args.local_rank),
                    'token_type_ids':
                    batch['token_type_ids'].to(args.local_rank),
                    'attention_mask':
                    batch['attention_mask'].to(args.local_rank),
                    'label_ids': batch['label_ids'].to(args.local_rank),
                    'domain_ids': batch['domain_ids'].to(args.local_rank),
                    'sample_weights':
                    batch['sample_weights'].to(args.local_rank),
                }
                label_ids = batch.pop('label_ids')
                if self.distill_stage == 'first':
                    # student_atts, student_reps, student_domain_rep
                    student_output = self._model(
                        input_dict,
                        is_student=True,
                        distill_stage=self.distill_stage)
                    with torch.no_grad():
                        # logits, teacher_atts, teacher_reps, teacher_domain_rep
                        teacher_output = self._teacher(input_dict,
                                                       is_student=False,
                                                       distill_stage='all')
                        teacher_probs = torch.softmax(teacher_output['logits'],
                                                      dim=-1)
                        label_onehots = torch.eye(
                            self.num_labels)[label_ids].to(args.local_rank)
                        grt_sample_weights = 1 / (torch.exp(
                            torch.sum(((teacher_probs - label_onehots) *
                                       label_onehots)**2,
                                      dim=-1)) + 1)

                    compute_loss_input = {
                        'distill_stage':
                        self.distill_stage,
                        'local_rank':
                        args.local_rank,
                        'label_ids':
                        label_ids,
                        'student_atts':
                        student_output['attentions'],
                        'student_reps':
                        student_output['sequence_output'],
                        'student_domain_rep':
                        student_output['domain_content_output'],
                        'teacher_atts':
                        teacher_output['attentions'],
                        'teacher_reps':
                        teacher_output['sequence_output'],
                        'teacher_domain_rep':
                        teacher_output['domain_content_output'],
                        'grt_sample_weights':
                        grt_sample_weights,
                        'sample_weights':
                        batch['sample_weights'].to(args.local_rank),
                        'domain_loss_weight':
                        self.domain_loss_weight,
                    }

                else:
                    student_output = self._model(
                        input_dict,
                        is_student=True,
                        distill_stage=self.distill_stage)
                    with torch.no_grad():
                        teacher_output = self._teacher(input_dict,
                                                       is_student=False,
                                                       distill_stage='second')
                    compute_loss_input = {
                        'distill_stage': self.distill_stage,
                        'local_rank': args.local_rank,
                        'label_ids': label_ids,
                        'student_logits': student_output['logits'],
                        'teacher_logits': teacher_output['logits'],
                        'T': self.T
                    }

                loss_dict = self.model_module.compute_loss(
                    **compute_loss_input)

                _loss = loss_dict['loss']
                if args.n_gpu > 1:
                    _loss = _loss.mean()
                if args.gradient_accumulation_steps > 1:
                    _loss = _loss / args.gradient_accumulation_steps
                _loss.backward()

                self.after_iter(_step, _epoch, loss_dict)
            self.after_epoch()
        print('Training Time: {}, rank {}, gsteps {}'.format(
            time.time() - self._start_time, args.rank, self._global_step))
        self.after_train()
