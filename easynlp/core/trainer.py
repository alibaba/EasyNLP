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

import contextlib
import math
import os
import sys
import time
from ast import literal_eval

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from ..utils import (exporter, get_args, get_dir_name, get_pretrain_model_path,
                     io, is_torchx_available)
from ..utils.logger import logger
from ..utils.statistics import Statistics
from .optimizers import get_optimizer

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    #EAS need this
    from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self, model, train_dataset, evaluator=None, **kwargs):
        self.args = get_args()
        # for ckbert contrast learning
        self.contrast_learning_flag = kwargs.get('contrast_learning_flag', False)
        # for save latent diffusion model
        if kwargs.get('user_defined_parameters',False):
            self.reset_model_state_flag = kwargs.get('user_defined_parameters',False).get('reset_model_state_flag',False)
        else:
            self.reset_model_state_flag=False
                    
        if self.args.use_torchacc == True and is_torchx_available() == False:
            raise ValueError('No TrochACC Running Environment!')
        if self.args.use_torchacc:
            import torchacc.torch_xla.core.xla_model as xm
            self._device = xm.xla_device()
            xm.set_replication(self._device, [self._device])

        if self.args.use_amp:
            if self.args.use_torchacc:
                from torchacc.torch_xla.amp import GradScaler
                self._scaler = GradScaler()
            else:
                self._scaler = torch.cuda.amp.GradScaler()
        self.optimizer_type = self.args.optimizer_type # add by ruihan.wjn
        self.max_grad_norm = self.args.max_grad_norm # add by ruihan.wjn
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None # add by ruihan.wjn
        self._train_loader = None
        self._start_epoch = 0
        self._start_global_step = 0
        self._start_time = time.time()
        self._current_loss = 0.
        self._current_epoch = self._start_epoch
        self.set_train_loader(train_dataset, self.args)
        self.set_model_and_optimizer(model, self.args)
        self.resume_from_ckpt(self.model_module, self.args)
        self.set_tensorboard()

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
        return self._optimizer.get_current_lr(self._lr_scheduler) if self._lr_scheduler else self._optimizer.get_current_lr()

    def set_model_and_optimizer(self, model, args):
        if self.args.use_torchacc:
            self._model = model.to(self._device)
        elif self.args.n_gpu == 1:
            self._device = self.args.local_rank
            self._model = model.to(self.args.local_rank)
        elif self.args.n_gpu > 1:
            self._device = self.args.local_rank
            self._model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.args.local_rank),
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True)
        else:
            logger.warn("Use CPU Training.")
            logger.warn("Make sure worker_gpu is set up correctly.")
            self._device = "cpu"
            self._model = model.to(self._device)

        # Build Optimizer
        self._optimizer, self._lr_scheduler = get_optimizer(
            optimizer_type=self.optimizer_type,
            learning_rate=args.learning_rate,
            warmup_proportion=args.warmup_proportion,
            max_grad_norm=self.max_grad_norm,
            named_parameters=list(self.model_module.named_parameters()),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_steps_per_epoch=len(self._train_loader),
            epoch_num=args.epoch_num,
            weight_decay=args.weight_decay,
        )

    def resume_from_ckpt(self, model_module, args):
        if args.resume_from_checkpoint is None:
            return
        meta_file = args.resume_from_checkpoint + '.meta.bin'
        model_file = args.resume_from_checkpoint + '.bin'
        if 'oss://' in args.resume_from_checkpoint:
            local_file = 'easynlp_resume_pytorch_model.meta.bin'
            io.download(meta_file, local_file)
            meta_file = local_file

            local_file = 'easynlp_resume_pytorch_model.bin'
            io.download(model_file, local_file)
            model_file = local_file

        with io.open(meta_file, 'rb') as f:
            meta_data = torch.load(f, map_location='cpu')
        self._start_epoch = meta_data['epoch']
        self._start_global_step = meta_data['global_step'] + 1
        self._optimizer.load_state_dict(meta_data['optimizer'])
        try:
            self._lr_scheduler.load_state_dict(meta_data['scheduler']) # add by ruihan.wjn
        except:
            self._lr_scheduler = None
        logger.info('Resume from checkpoint {}'.format(
            args.resume_from_checkpoint))
        logger.info('Start epoch {}'.format(self._start_epoch))
        logger.info('Start step {}'.format(self._start_global_step))
        logger.info('Start learning rate {:.6f}'.format(
                self._optimizer.get_current_lr(self._lr_scheduler) if self._lr_scheduler else self._optimizer.get_current_lr()
            )
        )
        with io.open(model_file, 'rb') as f:
            model_module.load_state_dict(torch.load(f, map_location='cpu'))
        logger.info('Resume checkpoint Done'.format(
            args.resume_from_checkpoint))

    def set_tensorboard(self):
        args = self.args
        if not args.is_master_node:
            return
        logger.info('=' * 10 + ' Initializing Tensorboard ' + '=' * 10)
        if 'oss://' in args.checkpoint_dir:
            self.tensorboard = SummaryWriter(
                log_dir=os.path.join('./easynlp_tensorboard'))
        else:
            self.tensorboard = SummaryWriter(
                log_dir=os.path.join(args.checkpoint_dir, 'log'))
        self.tensorboard.add_text(tag='config/training',
                                  text_string=str(self.args),
                                  global_step=0)

        self.tensorboard.add_text(
            tag='config/model_config',
            text_string=self.model_module.config.to_json_string(),
            global_step=0)

    def set_train_loader(self, train_dataset, args):

        if args.read_odps:
            train_sampler = None
        else:
            if self.args.use_torchacc:
                import torchacc.torch_xla.core.xla_model as xm
                if xm.xrt_world_size() > 1:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset,
                        num_replicas=xm.xrt_world_size(),
                        rank=xm.get_ordinal(),
                        shuffle=True)
                else:
                    train_sampler = None
            elif args.n_gpu <= 1:
                train_sampler = RandomSampler(train_dataset)
            else:
                train_sampler = DistributedSampler(train_dataset)

        if getattr(train_dataset, 'batch_fn', None) is not None:
            self._train_loader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=args.micro_batch_size,
                                            collate_fn=train_dataset.batch_fn,
                                            num_workers=self.args.data_threads)
        else:
            self._train_loader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=args.micro_batch_size,
                                            num_workers=self.args.data_threads)
        if self.args.use_torchacc:
            import torchacc.torch_xla.distributed.parallel_loader as pl
            self._train_loader = pl.MpDeviceLoader(self._train_loader,
                                                   self._device)

    def log_train_infos(self):
        args = self.args
        logger.info('=' * 10 + ' Training Start ' + '=' * 10 + '\n')
        logger.info('  Num of GPUs (all)       = %d', args.n_gpu)
        if args.n_gpu > 0:
            n_tr_samples = len(self._train_loader.dataset
                               ) * args.n_gpu if args.read_odps else len(
                                   self._train_loader.dataset)
            n_tr_batch_size = args.micro_batch_size * args.n_gpu * args.gradient_accumulation_steps
        else:
            n_tr_samples = len(
                self._train_loader.dataset) if args.read_odps else len(
                    self._train_loader.dataset)
            n_tr_batch_size = args.micro_batch_size * args.gradient_accumulation_steps
        n_tr_batch_no = len(self._train_loader.dataset
                            ) / args.micro_batch_size * args.epoch_num

        logger.info('  Num dataset examples    = %d',
                    len(self._train_loader.dataset))
        logger.info('  Num training examples   = %d', n_tr_samples)
        if self.evaluator is not None:
            logger.info('  Num validation examples = %d',
                        len(self.evaluator.valid_loader.dataset))
        logger.info('  Train. batch size       = %d', n_tr_batch_size)
        logger.info('  Train. micro batch size = %d', args.micro_batch_size)
        logger.info('  Train. batch no.        = %d', n_tr_batch_no)
        logger.info('  Evaluation batch size   = %d', args.micro_batch_size)
        # total_training_steps = self._optimizer.total_training_steps
        total_training_steps = int(
            math.ceil(
                len(self._train_loader) / args.gradient_accumulation_steps *
                args.epoch_num))
        self.total_training_steps = total_training_steps
        logger.info('  Total training steps    = %d', total_training_steps)
        logger.info('  Sequence length         = %s',
                    str(args.sequence_length))
        logger.info('  Saving steps            = %s',
                    str(args.save_checkpoint_steps))
        logger.info('  Distributed_backend     = %s',
                    str(args.distributed_backend))

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

        logger.info('=' * 10 + ' Model Config ' + '=' * 10)
        logger.info(self.model_module.config.to_json_string())

    def before_epoch(self, _epoch):
        args = self.args
        self._current_epoch = _epoch
        if args.n_gpu > 1:
            torch.distributed.barrier()
        self._model.train()
        self._epoch_tr_loss = 0.0
        self._epoch_n_tr_steps = 0.0
        if args.is_master_node:
            self._epoch_stats = Statistics(
                epoch_num=int(args.epoch_num),
                total_training_steps=self.total_training_steps)
            # total_training_steps=self._optimizer.total_training_steps)

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def autocast_context_manager(self):
        if self.args.use_amp:
            from torch.cuda.amp import autocast
            ctx_manager = autocast()
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (
                3, 7) else contextlib.suppress()

        return ctx_manager

    def optimizer_step(self):
        if self.args.use_torchacc:
            import torchacc.torch_xla.core.xla_model as xm
            if xm.xrt_world_size() > 1:
                gradients = xm._fetch_gradients(self._optimizer)
                xm.all_reduce('sum',
                              gradients,
                              scale=1.0 / xm.xrt_world_size())

        if self._lr_scheduler:
            # If use AdamW, it should explicit use clip grad
            if hasattr(self._optimizer, "clip_grad_norm"):
                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                self._optimizer.clip_grad_norm(self.max_grad_norm)
            elif hasattr(self.model_module, "clip_grad_norm_"):
                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                self.model_module.clip_grad_norm_(self.max_grad_norm)
            else:
                # Revert to normal clipping otherwise, handling Apex or full precision
                torch.nn.utils.clip_grad_norm_(self.model_module.parameters(), self.max_grad_norm)

        if self.args.use_amp:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            if self._lr_scheduler:
                # If use AdamW, it should explicit perform self._optimizer.step()
                # If not, it means use bertadam, which consists of inherent schedular.
                self._optimizer.step()
        if self._lr_scheduler:
            self._lr_scheduler.step()
        self._optimizer.zero_grad()

    def after_iter(self, _step, _epoch, loss_dict):
        args = self.args

        self.pred_loss = loss_dict['loss'].item()
        self._epoch_tr_loss += self.pred_loss
        self._epoch_n_tr_steps += 1
        if (_step + 1) % args.gradient_accumulation_steps == 0:
            self.optimizer_step()
            self._global_step += 1

            if not args.is_master_node:
                return
            self._epoch_stats.update(loss_dict)
            if self._global_step == 0 or (self._global_step +
                                          1) % args.logging_steps == 0:
                self._epoch_stats.output(self._global_step + 1, _epoch,
                                         self.learning_rate)
            self._epoch_stats.log_tensorboard(writer=self.tensorboard,
                                              learning_rate=self.learning_rate,
                                              current_loss=self.pred_loss,
                                              global_step=self._global_step,
                                              output_dir=os.path.join(
                                                  args.checkpoint_dir, 'log'))

            if args.save_checkpoint_steps and (
                    self._global_step + 1) % args.save_checkpoint_steps == 0:
                if args.save_all_checkpoints:
                    self.save_checkpoint()
                if self.evaluator is not None:
                    logger.info(
                        f'========== Evaluation at global step {self._global_step + 1} =========='
                    )
                    self._eval_scores = self.evaluator.evaluate(
                        model=self.model_module)
                    
                    if self._eval_scores[0][
                            1] > self.evaluator.best_valid_score:
                        logger.info(
                            'Saving best model to %s...' % os.path.join(
                                args.checkpoint_dir, 'pytorch_model.bin'))
                        self.save_checkpoint(save_best=True)
                        self.evaluator.best_valid_score = self._eval_scores[0][
                            1]
                        
                    logger.info('Best score: {}'.format(
                        self.evaluator.best_valid_score))
                    logger.info('Learning rate: {:.8f}'.format(
                        self._optimizer.get_current_lr(
                            self._lr_scheduler) if self._lr_scheduler else self._optimizer.get_current_lr()
                    ))
                    self._epoch_stats.log_tensorboard(
                        writer=self.tensorboard,
                        learning_rate=self.learning_rate,
                        eval_scores=self._eval_scores,
                        global_step=self._global_step,
                        is_training=False,
                        output_dir=os.path.join(args.checkpoint_dir, 'log'))

    def after_train(self):
        args = self.args
        # Save last checkpoint if needed
        if not args.is_master_node:
            return

        if args.save_checkpoint_steps is None:
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
        self.tensorboard.close()
        logger.info('Training Time: {}'.format(time.time() - self._start_time))

    def save_checkpoint(self, save_best=False):
        if not self.args.is_master_node:
            return

        exporter.export_train_config(
            saved_path=os.path.join(self.args.checkpoint_dir,
                                    'train_config.json'),
            vocab_dir=get_dir_name(self.args.checkpoint_dir),
            label_enumerate_values=self._train_loader.dataset.
            label_enumerate_values,
            model_config=self.model_module.config,
            cfg=self.args)

        exporter.export_label_mapping(
            saved_path=os.path.join(self.args.checkpoint_dir,
                                    'label_mapping.json'),
            label_enumerate_values=self._train_loader.dataset.
            label_enumerate_values)

        # Save config.json
        output_config_file = os.path.join(self.args.checkpoint_dir,
                                          'config.json')
        with io.open(output_config_file, 'w') as f:
            f.write(self.model_module.config.to_json_string())

        if self.args.pretrained_model_name_or_path is not None: 
            # Save vocab.txt
            if os.path.exists(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'vocab.txt')):
                io.copy(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'vocab.txt'),
                    os.path.join(get_dir_name(self.args.checkpoint_dir),
                                'vocab.txt'))
            # Save vocab.json
            elif os.path.exists(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'vocab.json')):
                io.copy(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'vocab.json'),
                    os.path.join(get_dir_name(self.args.checkpoint_dir),
                                'vocab.json'))
            # Save tokenizer.json
            elif os.path.exists(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'tokenizer.json')):
                io.copy(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'tokenizer.json'),
                    os.path.join(get_dir_name(self.args.checkpoint_dir),
                                'tokenizer.json'))
            else:
                raise FileNotFoundError

            # Save spiece.model
            spiece_path = os.path.join(
                get_dir_name(
                    get_pretrain_model_path(
                        self.args.pretrained_model_name_or_path,
                        disable_auto_download=True)), 'spiece.model')
            if os.path.exists(spiece_path):
                io.copy(
                    spiece_path,
                    os.path.join(get_dir_name(self.args.checkpoint_dir),
                                'spiece.model'))
            # save super-resolution model
            if  os.path.exists(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'RRDB_ESRGAN_x4.pth')):
                io.copy(
                    os.path.join(
                        get_dir_name(
                            get_pretrain_model_path(
                                self.args.pretrained_model_name_or_path,
                                disable_auto_download=True)), 'RRDB_ESRGAN_x4.pth'),
                    os.path.join(get_dir_name(self.args.checkpoint_dir),
                                'RRDB_ESRGAN_x4.pth'))
        # Save the model
        model_to_save_prefix = 'pytorch_model' if save_best else 'pytorch_model_step_%d' % (
            self._global_step + 1)
        with io.open(os.path.join(self.args.checkpoint_dir, model_to_save_prefix + '.bin'), 'wb') \
                as output_model_file:
            if self.reset_model_state_flag:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in self.model_module.state_dict().items():
                    name = k[6:]   # remove `model.`
                    new_state_dict[name] = v
                torch.save(new_state_dict, output_model_file)
            else:    
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

        """
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
            """

        # This is a hack
        if torch.cuda.is_available():
            torch.cuda.set_device(self.args.local_rank)

    def contrast_learning_process(self, positive_negative_examples: torch.Tensor) -> torch.Tensor:
        # compute the exapmle emmbding
        original_size = positive_negative_examples.size()
        positive_negative_examples_inputs = positive_negative_examples.view(-1, original_size[-1])
        input_inter = torch.sum(positive_negative_examples_inputs, dim=-1)
        input_nozero = torch.nonzero(input_inter)
        true_positive_negative_examples_inputs = positive_negative_examples_inputs[input_nozero]
    
        positive_negative_example_results = self._model({
            "input_ids": true_positive_negative_examples_inputs.squeeze(1)
        })
        
        positive_negative_example_hidden_states = positive_negative_example_results['hidden_states']
        output_size = positive_negative_example_hidden_states.size()
        positive_negative_examples_inputs = positive_negative_examples_inputs.unsqueeze(-1).repeat(1, 1, output_size[-1]).float()
        positive_negative_examples_inputs[input_nozero.squeeze(-1)] = positive_negative_example_hidden_states
        positive_negative_examples_results_new = positive_negative_examples_inputs.view(original_size[0], original_size[1], original_size[2], original_size[3], output_size[-1])
        return positive_negative_examples_results_new
    
    def train(self):
        self.log_train_infos()
        args = self.args
        with contextlib.suppress():
            for _epoch in range(self._start_epoch, int(args.epoch_num)):
                self.before_epoch(_epoch)
                start_time = time.time()
                loss_total = 0
                pn_loss_total = 0
                for _step, batch in enumerate(self._train_loader):
                    if self._global_step + 1 < self._start_global_step:
                        if (_step + 1) % args.gradient_accumulation_steps == 0:
                            self._global_step += 1
                        continue
                    self.before_iter()

                    if not self.args.use_torchacc:
                        batch = {
                            key: val.to(self._device) if isinstance(
                                val, torch.Tensor) else val
                            for key, val in batch.items()
                        }

                    label_ids = batch.pop('label_ids', None)
                    positive_negative_examples = batch.pop('positive_negative_examples', None)
                    with self.autocast_context_manager():
                        if label_ids is not None:
                            forward_outputs = self._model(batch)
                            # for ckbert contrast learning
                            if self.contrast_learning_flag:
                                positive_negative_examples_results_new = self.contrast_learning_process(positive_negative_examples)
                        else:
                            forward_outputs, label_ids = self._model(batch)
                        if batch.get('insert_know_labels') is not None:
                            loss_dict = self.model_module.compute_loss(
                                forward_outputs, label_ids,
                                batch.get('insert_know_labels'))
                        elif self.contrast_learning_flag:
                            loss_dict = self.model_module.compute_loss(
                                forward_outputs, 
                                label_ids,
                                constrast_learning_flag = self.contrast_learning_flag,
                                positive_negative_results = positive_negative_examples_results_new
                            )
                        else:
                            loss_dict = self.model_module.compute_loss(
                                forward_outputs, label_ids)

                    _loss = loss_dict['loss']
                    if self.contrast_learning_flag:
                        _loss = loss_dict['loss'] + loss_dict['cl_loss']
                        
                    if args.n_gpu > 1:
                        _loss = _loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        _loss = _loss / args.gradient_accumulation_steps

                    if self.args.use_amp:
                        self._scaler.scale(_loss).backward()
                    else:
                        _loss.backward()

                    if self.contrast_learning_flag:
                        loss_total += _loss
                        pn_loss_total += loss_dict['cl_loss']
                        if _step % 10 == 0:
                            logger.info(f"total loss: {loss_total/10}\t knowledge loss: {(loss_total - pn_loss_total)/10}\t contrast loss: {pn_loss_total/10}")
                            loss_total = 0
                            pn_loss_total = 0
                        
                    self.after_iter(_step, _epoch, loss_dict)

                end_time = time.time()
                self.after_epoch()
        print('Training Time: {}, rank {}, gsteps {}'.format(
            time.time() - self._start_time, args.rank, self._global_step))
        self.after_train()