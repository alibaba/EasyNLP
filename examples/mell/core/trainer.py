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
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from evaluator import Evaluator
from optimizers import get_optimizer
from layers.base import WEIGHTS_NAME, CONFIG_NAME
from utils import exporter, io, get_dir_name
from utils.logger import logger
from utils.statistics import Statistics


class Trainer(object):
    def __init__(self, model, train_dataset, valid_dataset, cfg, evaluator=None):
        self.cfg = cfg
        self._model = None
        self._optimizer = None
        self._train_loader = None
        self._valied_loader = None
        self._start_epoch = 0
        self._start_global_step = 0
        self._start_time = time.time()
        self._current_loss = 0.
        self._eval_scores = None
        self._best_valid_score = float('-inf')
        self._current_epoch = self._start_epoch

        self.set_evaluator(evaluator, valid_dataset.eval_metrics)
        self.set_data_loader(train_dataset, valid_dataset, cfg)
        self.set_model_and_optimizer(model, cfg)
        self.resume_from_ckpt(self.model_module, cfg)
        self.set_tensorboard()

        self._global_step = self._start_epoch * len(self._train_loader)

    @property
    def model_module(self):
        if self._model is None:
            return self._model
        # space left for apex/deepspeed

        return self._model.module if hasattr(self._model, 'module') else self._model

    @property
    def learning_rate(self):
        return self._optimizer.get_current_lr()

    def set_model_and_optimizer(self, model, cfg):
        self._model = model.to(self.cfg.local_rank)
        if self.cfg.n_gpu > 1:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model, device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=True)

        # Build Optimizer
        self._optimizer = get_optimizer(optimizer_type="adam",
                                        learning_rate=cfg.learning_rate,
                                        warmup_proportion=cfg.warmup_proportion,
                                        max_grad_norm=cfg.max_grad_norm,
                                        named_parameters=list(self.model_module.named_parameters()),
                                        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                                        num_steps_per_epoch=len(self._train_loader),
                                        epoch_num=cfg.epoch_num)
        # space left for apex/deepspeed

    def resume_from_ckpt(self, model_module, cfg):
        if cfg.resume_from_checkpoint is None:
            return
        meta_file = cfg.resume_from_checkpoint + ".meta.bin"
        model_file = cfg.resume_from_checkpoint + ".bin"
        if "oss::" in cfg.resume_from_checkpoint:
            local_file = "easytexminer_resume_pytorch_model.meta.bin"
            io.download(model_file, local_file)
            meta_file = local_file

            local_file = "easytexminer_resume_pytorch_model.bin"
            io.download(model_file, local_file)
            model_file = local_file

        with io.open(meta_file, "rb") as f:
            meta_data = torch.load(f, map_location='cpu')
        self._start_epoch = meta_data["epoch"]
        self._start_global_step = meta_data["global_step"] + 1
        self._optimizer.load_state_dict(meta_data['optimizer'])

        logger.info("Resume from checkpoint {}".format(cfg.resume_from_checkpoint))
        logger.info("Start epoch {}".format(self._start_epoch))
        logger.info("Start step {}".format(self._start_global_step))
        logger.info("Start learning rate {:.6f}".format(self._optimizer.get_current_lr()))
        with io.open(model_file, "rb") as f:
            model_module.load_state_dict(torch.load(f, map_location='cpu'))
        logger.info("Resume checkpoint Done".format(cfg.resume_from_checkpoint))

    def set_tensorboard(self):
        cfg = self.cfg
        if not cfg.is_master_node:
            return
        logger.info("=" * 10 + " Initializing Tensorboard " + "=" * 10)
        if "oss://" in cfg.checkpoint_dir:
            self.tensorboard = SummaryWriter(log_dir=os.path.join("./easytexminer_tensorboard"))
        else:
            self.tensorboard = SummaryWriter(log_dir=os.path.join(cfg.checkpoint_dir, "log"))
        self.tensorboard.add_text(tag="config/training", text_string=str(self.cfg), global_step=0)
        self.tensorboard.add_text(tag="config/model_arch",
                                  text_string=self.model_module.arch, global_step=0)

    def set_evaluator(self, evaluator=None, eval_metrics=None):
        if evaluator is None:
            self.evaluator = Evaluator(metrics=eval_metrics)
        else:
            self.evaluator = evaluator

    def set_data_loader(self, train_dataset, valid_dataset, cfg):
        if cfg.read_odps:
            train_sampler = None
        else:
            train_sampler = RandomSampler if cfg.n_gpu <= 1 else DistributedSampler

        self._train_loader = DataLoader(train_dataset,
                                        sampler=train_sampler(train_dataset) if train_sampler else None,
                                        batch_size=cfg.train_batch_size,
                                        collate_fn=train_dataset.batch_fn)

        self._valid_loader = DataLoader(valid_dataset,
                                        batch_size=cfg.eval_batch_size,
                                        shuffle=False,
                                        collate_fn=valid_dataset.batch_fn)

    def log_train_infos(self):
        cfg = self.cfg

        logger.info("=" * 10 + " Training Start " + "=" * 10 + "\n")
        logger.info("  Num of GPUs   = %d", cfg.n_gpu)
        n_tr_samples = len(self._train_loader.dataset) * cfg.n_gpu if cfg.read_odps else len(self._train_loader.dataset)
        logger.info("  Num training examples   = %d", n_tr_samples)
        logger.info("  Num validation examples = %d", len(self._valid_loader.dataset))
        logger.info("  Training batch size     = %d",
                    cfg.train_batch_size * cfg.n_gpu * cfg.gradient_accumulation_steps)
        logger.info("  Evaluation batch size   = %d", cfg.eval_batch_size)
        total_training_steps = self._optimizer.total_training_steps
        logger.info("  Total training steps    = %d", total_training_steps)
        logger.info("  Saving steps            = %s", str(cfg.save_checkpoint_steps))

        model_num_params = sum([p.nelement() for n, p in self.model_module.named_parameters()])
        trainable_num_params = sum([p.nelement() for n, p in self.model_module.named_parameters() if p.requires_grad])
        logger.info("  num model parameters  = %s" % format(model_num_params, ","))
        logger.info("  num trainable parameters  = %s" % format(trainable_num_params, ","))
        logger.info("\n")

        logger.info("=" * 10 + " Model Arch " + "=" * 10)
        logger.info(self.model_module.arch)

    def before_epoch(self, _epoch):
        cfg = self.cfg
        self._current_epoch = _epoch
        if cfg.n_gpu > 1:
            torch.distributed.barrier()
        self._model.train()
        self._epoch_tr_loss = 0.0
        self._epoch_n_tr_steps = 0.0
        if cfg.is_master_node:
            self._epoch_stats = Statistics(epoch_num=int(cfg.epoch_num),
                                           total_training_steps=self._optimizer.total_training_steps)

    def after_epoch(self):
        pass

    def before_iter(self):
        pass

    def optimizer_step(self):
        self._optimizer.step()
        self._optimizer.zero_grad()

    def after_iter(self, _step, _epoch, loss_dict):
        cfg = self.cfg

        self.pred_loss = loss_dict["loss"].item()
        self._epoch_tr_loss += self.pred_loss
        self._epoch_n_tr_steps += 1

        if (_step + 1) % cfg.gradient_accumulation_steps == 0:
            self.optimizer_step()
            self._global_step += 1

            if not cfg.is_master_node:
                return
            self._epoch_stats.update(loss_dict)
            if self._global_step == 0 or (self._global_step + 1) % cfg.logging_steps == 0:
                self._epoch_stats.output(self._global_step + 1, _epoch, self.learning_rate)
            self._epoch_stats.log_tensorboard(writer=self.tensorboard,
                                              learning_rate=self.learning_rate,
                                              current_loss=self.pred_loss,
                                              global_step=self._global_step,
                                              output_dir=os.path.join(cfg.checkpoint_dir, "log"))

            if cfg.save_checkpoint_steps and (self._global_step + 1) % cfg.save_checkpoint_steps == 0:
                print()
                if cfg.save_all_checkpoints:
                    self.save_checkpoint()
                self._eval_scores = self.evaluator.evaluate(
                    model=self._model, valid_loader=self._valid_loader)
                if self._eval_scores[0][1] > self._best_valid_score:
                    logger.info("Saving best model to %s..." % os.path.join(cfg.checkpoint_dir, WEIGHTS_NAME))
                    self.save_checkpoint(save_best=True)
                    self._best_valid_score = self._eval_scores[0][1]
                logger.info("Best score: {}".format(self._best_valid_score))
                logger.info("Learning rate: {:.8f}".format(self._optimizer.get_current_lr()))
                logger.info("")
                self._epoch_stats.log_tensorboard(writer=self.tensorboard,
                                                  learning_rate=self.learning_rate,
                                                  eval_scores=self._eval_scores,
                                                  global_step=self._global_step,
                                                  is_training=False,
                                                  output_dir=os.path.join(cfg.checkpoint_dir, "log"))

    def after_train(self):
        cfg = self.cfg
        # Save last checkpoint if needed
        if not cfg.is_master_node:
            return
        if cfg.save_checkpoint_steps is None:
            logger.info("Saving best model to %s..." % os.path.join(cfg.checkpoint_dir, WEIGHTS_NAME))
            self.save_checkpoint(save_best=True)
        else:
            self._eval_scores = self.evaluator.evaluate(
                model=self._model, valid_loader=self._valid_loader)
            if self._eval_scores[0][1] > self._best_valid_score:
                logger.info("Saving best model to %s..." % os.path.join(cfg.checkpoint_dir, WEIGHTS_NAME))
                self.save_checkpoint(save_best=True)
                self._best_valid_score = self._eval_scores[0][1]
            logger.info("Best score: {}".format(self._best_valid_score))
        self.tensorboard.close()
        logger.info("Training Time: {}".format(time.time() - self._start_time))

    def save_checkpoint(self, save_best=False):
        if not self.cfg.is_master_node:
            return

        # Save config.json
        output_config_file = os.path.join(self.cfg.checkpoint_dir, CONFIG_NAME)
        with io.open(output_config_file, "w") as f:
            f.write(self.model_module.arch)

        # Save vocab.txt
        if self.cfg.pretrain_model_name_or_path is not None:
            io.copy(os.path.join(get_dir_name(self.cfg.pretrain_model_name_or_path), "vocab.txt"),
                    os.path.join(get_dir_name(self.cfg.checkpoint_dir), "vocab.txt"))

        # Save the model
        model_to_save_prefix = "pytorch_model" if save_best else "pytorch_model_step_%d" % (self._global_step + 1)

        with io.open(os.path.join(self.cfg.checkpoint_dir, model_to_save_prefix + ".bin"), "wb") \
                as output_model_file:
            torch.save(self.model_module.state_dict(), output_model_file)

        meta_data = {
            "epoch": self._current_epoch,
            "global_step": self._global_step,
            "optimizer": self._optimizer.state_dict()
        }

        with io.open(os.path.join(self.cfg.checkpoint_dir, model_to_save_prefix + ".meta.bin"), "wb") \
                as output_model_file:
            torch.save(meta_data, output_model_file)

        if not save_best:
            return

        if hasattr(self.model_module, "model_name"):
            # If the student is pre-defined EasyTransfer AppZoo model
            # Save train_config.json, model.ckpt.* for EasyTransfer
            logger.info("Export tensorflow checkpoint (%s format) to %s" % (
                self.cfg.export_tf_checkpoint_type,
                os.path.join(get_dir_name(self.cfg.checkpoint_dir), "model.ckpt")))
            exporter.export_easytransfer_train_config(
                saved_path=os.path.join(self.cfg.checkpoint_dir, "train_config.json"),
                vocab_dir=get_dir_name(self.cfg.checkpoint_dir),
                label_enumerate_values=self._valid_loader.dataset.label_enumerate_values,
                sequence_length=self.cfg.sequence_length,
                model_name=self.model_module.model_name,
                extra_model_params=self.model_module.extra_model_params)

            if self.cfg.export_tf_checkpoint_type == "easytransfer":
                exporter.export_pytorch_checkpoint_to_tf(
                    model=self.model_module,
                    ckpt_dir=get_dir_name(self.cfg.checkpoint_dir),
                    bert_output_prefix="bert_pre_trained_model",
                    appended_val_map=(("classifier", "app/ez_dense"),),
                    appended_tensors_to_transpose=("classifier.weight",))
            elif self.cfg.export_tf_checkpoint_type == "google":
                exporter.export_pytorch_checkpoint_to_tf(
                    model=self.model_module,
                    ckpt_dir=get_dir_name(self.cfg.checkpoint_dir),
                    bert_output_prefix="",
                    appended_val_map=(("classifier.weight", "output_weights"),
                                      ("classifier.bias", "output_bias")),
                    appended_tensors_to_transpose=())
            else:
                raise RuntimeError("Invalid export_tf_checkpoint_type %s" % self.cfg.export_tf_checkpoint_type)
        # This is a hack
        torch.cuda.set_device(self.cfg.local_rank)

    def train(self):
        self.log_train_infos()
        cfg = self.cfg

        for _epoch in range(self._start_epoch, int(cfg.epoch_num)):
            self.before_epoch(_epoch)
            for _step, batch in enumerate(self._train_loader):
                print('running step: {}'.format(_step))
                if self._global_step + 1 < self._start_global_step:
                    if (_step + 1) % cfg.gradient_accumulation_steps == 0:
                        self._global_step += 1
                    continue
                self.before_iter()
                batch = {key: val.to(cfg.local_rank) if isinstance(val, torch.Tensor) else val
                         for key, val in batch.items()}
                # model output logis
                model_outputs = self._model(batch)
                loss_dict = self.model_module.compute_loss(model_outputs, batch)

                _loss = loss_dict["loss"]
                if cfg.n_gpu > 1:
                    _loss = _loss.mean()
                if cfg.gradient_accumulation_steps > 1:
                    _loss = _loss / cfg.gradient_accumulation_steps

                _loss.backward()

                self.after_iter(_step, _epoch, loss_dict)
            self.after_epoch()
        print('running to here!!!!')
        self.after_train()


class DistillTrainer(Trainer):
    def __init__(self,
                 teacher_model,
                 student_model,
                 distiller,
                 train_dataset,
                 valid_dataset,
                 cfg,
                 evaluator=None):

        self.cfg = cfg
        self.teacher = teacher_model
        self.teacher = self.teacher.to(cfg.local_rank) if self.teacher else None
        student_model.init_from_teacher(self.teacher, cfg.student_init_strategy)
        self.distiller = distiller
        self.need_teacher_in_eval = "last_layer_mse" in train_dataset.eval_metrics

        super(DistillTrainer, self).__init__(student_model, train_dataset, valid_dataset,
                                             cfg, evaluator=evaluator)

        if cfg.is_master_node:
            if self.teacher is not None:
                self.tensorboard.add_text(tag="config/teacher_arch",
                                          text_string=str(self.teacher.module.arch
                                                          if hasattr(self.teacher, "module")
                                                          else self.teacher.arch), global_step=0)
            self.tensorboard.add_text(tag="config/kd_map",
                                      text_string=json.dumps(self.distiller.kd_map, indent=4), global_step=0)

    def log_train_infos(self):
        super(DistillTrainer, self).log_train_infos()

        if self.teacher is not None:
            logger.info("=" * 10 + " Teacher Arch " + "=" * 10)
            logger.info(self.teacher.module.arch if hasattr(self.teacher, "module") else self.teacher.arch)
        logger.info("=" * 10 + " Distiller KD Map " + "=" * 10)
        logger.info(json.dumps(self.distiller.kd_map, indent=4))

    def train(self):
        self.log_train_infos()
        cfg = self.cfg

        for _epoch in range(self._start_epoch, int(cfg.epoch_num)):
            self.before_epoch(_epoch)

            for _step, batch in enumerate(self._train_loader):
                if self._global_step + 1 < self._start_global_step:
                    if (_step + 1) % cfg.gradient_accumulation_steps == 0:
                        self._global_step += 1
                    continue
                self.before_iter()

                batch = {key: val.to(cfg.local_rank) if isinstance(val, torch.Tensor) else val
                         for key, val in batch.items()}
                if self.teacher is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher(batch)
                elif "teacher_logits" in batch:
                    teacher_outputs = {
                        "logits": batch["teacher_logits"]
                    }
                else:
                    teacher_outputs = {}
                student_outputs = self._model(batch)
                label_ids = batch["label_ids"] if "label_ids" in batch else None
                attention_mask = batch["input_mask"] if "input_mask" in batch else None
                kd_loss, kd_loss_dict = self.distiller(
                    teacher_outputs, student_outputs, label_ids, attention_mask)
                kd_loss_dict["loss"] = kd_loss

                if cfg.n_gpu > 1:
                    kd_loss = kd_loss.mean()
                if cfg.gradient_accumulation_steps > 1:
                    kd_loss = kd_loss / cfg.gradient_accumulation_steps

                kd_loss.backward()

                self.after_iter(_step, _epoch, kd_loss_dict)

            self.after_epoch()

        self.after_train()
