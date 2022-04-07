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

import os
import time
from ast import literal_eval

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..utils import (exporter, get_args, get_dir_name, get_pretrain_model_path,
                     io)
from ..utils.logger import logger
from ..utils.statistics import Statistics
from .optimizers import get_optimizer


class VanillaTrainer(object):
    def __init__(self, model, train_dataset, evaluator, **kwargs):
        self.args = get_args()
        self._model = None
        self._optimizer = None
        self._train_loader = None
        self._start_epoch = 0
        self._start_global_step = 0
        self._start_time = time.time()
        self._current_loss = 0.
        self._current_epoch = self._start_epoch

        print('log: set train loader')
        self.set_train_loader(train_dataset, self.args)

        print('log: set_model_and_optimizer')
        self.set_model_and_optimizer(model, self.args)
        print('log: resume_from_ckpt')
        self.resume_from_ckpt(self.model_module, self.args)

        # print('log: set_tensorboard')
        # self.set_tensorboard()
        self._global_step = self._start_epoch * len(self._train_loader)
        self.evaluator = evaluator

    @property
    def model_module(self):
        if self._model is None:
            return self._model

        return self._model.module if hasattr(self._model,
                                             'module') else self._model

    @property
    def learning_rate(self):
        return self._optimizer.get_current_lr()

    def set_model_and_optimizer(self, model, args):
        if self.args.n_gpu == 1:
            self._model = model.to(self.args.local_rank)
        elif self.args.n_gpu > 1:
            self._model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.args.local_rank),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True)
        else:
            raise Exception('CPU Training is not supported.')

        # # Build Optimizer
        # self._optimizer = get_optimizer(
        #     optimizer_type="adam",
        #     learning_rate=args.learning_rate,
        #     warmup_proportion=args.warmup_proportion,
        #     max_grad_norm=args.max_grad_norm,
        #     named_parameters=list(self.model_module.named_parameters()),
        #     gradient_accumulation_steps=args.gradient_accumulation_steps,
        #     num_steps_per_epoch=len(self._train_loader),
        #     epoch_num=args.epoch_num)
        print('log: set SGD')
        self._optimizer = torch.optim.SGD(self._model.parameters(),
                                          lr=args.learning_rate,
                                          momentum=0.9)

    def resume_from_ckpt(self, model_module, args):
        if args.resume_from_checkpoint is None:
            return
        meta_file = args.resume_from_checkpoint + '.meta.bin'
        model_file = args.resume_from_checkpoint + '.bin'
        if 'oss::' in args.resume_from_checkpoint:
            local_file = 'easynlp_resume_pytorch_model.meta.bin'
            io.download(model_file, local_file)
            meta_file = local_file

            local_file = 'easynlp_resume_pytorch_model.bin'
            io.download(model_file, local_file)
            model_file = local_file

        with io.open(meta_file, 'rb') as f:
            meta_data = torch.load(f, map_location='cpu')
        self._start_epoch = meta_data['epoch']
        self._start_global_step = meta_data['global_step'] + 1
        self._optimizer.load_state_dict(meta_data['optimizer'])

        logger.info('Resume from checkpoint {}'.format(
            args.resume_from_checkpoint))
        logger.info('Start epoch {}'.format(self._start_epoch))
        logger.info('Start step {}'.format(self._start_global_step))
        logger.info('Start learning rate {:.6f}'.format(
            self._optimizer.get_current_lr()))
        with io.open(model_file, 'rb') as f:
            model_module.load_state_dict(torch.load(f, map_location='cpu'))
        logger.info('Resume checkpoint Done'.format(
            args.resume_from_checkpoint))

    def set_train_loader(self, train_dataset, args):
        if args.read_odps:
            train_sampler = None
        else:
            train_sampler = RandomSampler if args.n_gpu <= 1 else DistributedSampler
            # train_sampler = SequentialSampler

        # Init dataloader to make sure at least one cpu core is loading data
        # Note: here num_worker=n_cpu
        if getattr(train_dataset, 'batch_fn', None) is not None:
            self._train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler(train_dataset)
                if train_sampler else None,
                batch_size=args.micro_batch_size,
                collate_fn=train_dataset.batch_fn,
                num_workers=max(args.n_cpu, 1))
        else:
            self._train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler(train_dataset)
                if train_sampler else None,
                batch_size=args.micro_batch_size,
                num_workers=max(args.n_cpu, 1))

    def log_train_infos(self):
        args = self.args

        logger.info('=' * 10 + ' Training Start ' + '=' * 10 + '\n')
        logger.info('  Num of GPUs (all)       = %d', args.n_gpu)
        logger.info('  Num of CPUs per worker  = %d', args.n_cpu)
        if args.n_gpu > 0:
            n_tr_samples = len(self._train_loader.dataset
                               ) * args.n_gpu if args.read_odps else len(
                                   self._train_loader.dataset)
            n_tr_batch_size = args.micro_batch_size * args.n_gpu * args.gradient_accumulation_steps
        else:
            n_tr_samples = len(self._train_loader.dataset)
            n_tr_batch_size = args.micro_batch_size * args.gradient_accumulation_steps
        n_tr_batch_no = len(self._train_loader.dataset
                            ) / args.micro_batch_size * args.epoch_num

        logger.info('  Num dataset examples    = %d',
                    len(self._train_loader.dataset))
        logger.info('  Num training examples   = %d', n_tr_samples)
        if self.evaluator is not None:
            logger.info('  Num validation examples = %d',
                        len(self.evaluator.valid_loader.dataset))
        logger.info('  Train. steps            = %d',
                    len(self._train_loader.dataset) / args.micro_batch_size)
        logger.info('  Train. batch size       = %d', n_tr_batch_size)
        logger.info('  Train. micro batch size = %d', args.micro_batch_size)
        logger.info('  Train. batch no.        = %d', n_tr_batch_no)
        logger.info('  Evaluation batch size   = %d', args.micro_batch_size)
        logger.info('  Sequence length         = %s',
                    str(args.sequence_length))
        logger.info('  Saving steps            = %s',
                    str(args.save_checkpoint_steps))
        logger.info('  Distributed_backend     = %s',
                    str(args.distributed_backend))
        logger.info('  Worker Count            = %s', str(args.worker_count))
        logger.info('  Worker CPU              = %s', str(args.worker_cpu))
        logger.info('  Worker data threads     = %s', str(args.data_threads))

        model_num_params = sum(
            [p.nelement() for n, p in self.model_module.named_parameters()])
        trainable_num_params = sum([
            p.nelement() for n, p in self.model_module.named_parameters()
            if p.requires_grad
        ])
        logger.info('  num model params        = %s' %
                    format(model_num_params, ','))
        logger.info('  num trainable params    = %s' %
                    format(trainable_num_params, ','))
        logger.info('\n')

    def optimizer_step(self):
        self._optimizer.step()
        self._optimizer.zero_grad()

    def after_train(self):
        args = self.args

        # Save last checkpoint if needed
        if not args.is_master_node:
            return

        if args.save_checkpoint_steps is not None:
            logger.info('Saving best model to %s...' %
                        os.path.join(args.checkpoint_dir, 'pytorch_model.bin'))
            self.save_checkpoint(save_best=True)
        elif self.evaluator is not None:
            self._eval_scores = self.evaluator.evaluate(
                model=self.model_module)
            if self._eval_scores[0][1] > self.evaluator.best_valid_score:
                logger.info(
                    'Saving best model to %s...' %
                    os.path.join(args.checkpoint_dir, 'pytorch_model.bin'))
                self.save_checkpoint(save_best=True)
                self.evaluator.best_valid_score = self._eval_scores[0][1]
            logger.info('Best score: {}'.format(
                self.evaluator.best_valid_score))

        # self.tensorboard.close()

        logger.info('Destroy Process Group.')
        torch.distributed.destroy_process_group()

    def save_checkpoint(self, save_best=False):
        if not self.args.is_master_node:
            return

        # Save the model
        model_to_save_prefix = 'pytorch_model' if save_best else 'pytorch_model_step_%d' % (
            self._global_step + 1)

        with io.open(os.path.join(self.args.checkpoint_dir, model_to_save_prefix + '.bin'), 'wb') \
                as output_model_file:
            torch.save(self.model_module.state_dict(), output_model_file)

        meta_data = {
            'epoch': self._current_epoch,
            'global_step': self._global_step,
            'optimizer': self._optimizer.state_dict()
        }

        with io.open(os.path.join(self.args.checkpoint_dir, model_to_save_prefix + '.meta.bin'), 'wb') \
                as output_model_file:
            torch.save(meta_data, output_model_file)

        if not save_best:
            return

        if self.args.export_tf_checkpoint_type != 'none' and hasattr(
                self.model_module, 'model_name'):
            # If the student is pre-defined EasyTransfer AppZoo model
            # Save train_config.json, model.ckpt.* for EasyTransfer
            logger.info('Export tensorflow checkpoint (%s format) to %s' %
                        (self.args.export_tf_checkpoint_type,
                         os.path.join(get_dir_name(self.args.checkpoint_dir),
                                      'model.ckpt')))

            if self.args.export_tf_checkpoint_type == 'easytransfer':
                exporter.export_pytorch_checkpoint_to_tf(
                    model=self.model_module,
                    ckpt_dir=get_dir_name(self.args.checkpoint_dir),
                    bert_output_prefix='bert_pre_trained_model',
                    appended_val_map=(('classifier', 'app/ez_dense'), ),
                    appended_tensors_to_transpose=('classifier.weight', ))
            elif self.args.export_tf_checkpoint_type == 'google':
                exporter.export_pytorch_checkpoint_to_tf(
                    model=self.model_module,
                    ckpt_dir=get_dir_name(self.args.checkpoint_dir),
                    bert_output_prefix='',
                    appended_val_map=(('classifier.weight', 'output_weights'),
                                      ('classifier.bias', 'output_bias')),
                    appended_tensors_to_transpose=())
            else:
                raise RuntimeError('Invalid export_tf_checkpoint_type %s' %
                                   self.args.export_tf_checkpoint_type)
        # This is a hack
        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)

    def train(self):
        print('log: train loop...')
        self.log_train_infos()
        args = self.args
        for _epoch in range(self._start_epoch, int(args.epoch_num)):
            if args.n_gpu > 1:
                torch.distributed.barrier()

            running_loss = 0.0
            for _step, batch in enumerate(self._train_loader):
                if self._global_step % 100 == 0:
                    print(
                        'Worker step %4d, batch %4d, rank %d, l_rank %d, master %5s, loss %f'
                        %
                        (self._global_step, _step, args.rank, args.local_rank,
                         args.is_master_node, running_loss / 100))
                    running_loss = 0.0
                self._global_step += 1

                # compute loss
                if type(batch) is dict:
                    batch = {
                        key: val.to(args.local_rank) if isinstance(
                            val, torch.Tensor) else val
                        for key, val in batch.items()
                    }

                    label_ids = batch.pop('label_ids')
                    # loss_dict = self.model_module.compute_loss(batch.pop("input_ids"), label_ids)

                    forward_outputs = self._model(batch)
                    loss_dict = self.model_module.compute_loss(
                        forward_outputs, label_ids)
                else:
                    # call user-defined loss function
                    loss_dict = self.model_module.compute_loss(
                        batch[0], batch[1])

                _loss = loss_dict['loss']
                if args.n_gpu > 1:
                    _loss = _loss.mean()

                _loss.backward()
                running_loss += _loss.item()
                self.optimizer_step()

        print('Training Time: {}, rank {}, gsteps {}'.format(
            time.time() - self._start_time, args.rank, self._global_step))
        # save ckpt
        self.after_train()
        print('Finish training.')
