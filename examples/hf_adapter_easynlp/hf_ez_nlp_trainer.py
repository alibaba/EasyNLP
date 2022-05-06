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
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    #EAS need this
    from tensorboardX import SummaryWriter
from easynlp.core.optimizers import get_optimizer
from easynlp.utils import exporter, io, get_dir_name, get_pretrain_model_path
from easynlp.utils.logger import logger
from easynlp.utils.statistics import Statistics
from easynlp.utils import get_args
from easynlp.core import Trainer
class HfTrainer(Trainer):

    def __init__(self, model, train_dataset, evaluator, **kwargs):
        self.args = get_args()
        self._model = None
        self.optimizer_type = self.args.optimizer_type
        self.max_grad_norm = self.args.max_grad_norm
        self._optimizer = None
        self._train_loader = None
        self._start_epoch = 0
        self._start_global_step = 0
        self._start_time = time.time()
        self._current_loss = 0.
        self._lr_scheduler = None
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

        return self._model.module if hasattr(self._model, 'module') else self._model

    @property
    def learning_rate(self):
        return self._optimizer.get_current_lr()
        
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
                forward_outputs, label_ids, _ = self._model.forward_repre(self._model, args, deepcopy(batch))
                loss_dict = self._model.compute_loss(forward_outputs, label_ids)                    
                _loss = loss_dict["loss"]
                if args.n_gpu > 1:
                    _loss = _loss.mean()
                if args.gradient_accumulation_steps > 1:
                    _loss = _loss / args.gradient_accumulation_steps

                _loss.backward()

                self.after_iter(_step, _epoch, loss_dict)

            self.after_epoch()

        self.after_train()
