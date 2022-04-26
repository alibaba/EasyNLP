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

from __future__ import division
import os
import time

from .logger import logger
from . import io


class Statistics(object):
    """
    Accumulator for loss statistics.
    """

    def __init__(self, epoch_num, total_training_steps):
        self.epoch_num = epoch_num
        self.total_training_steps = total_training_steps
        self.n_tr_steps = 0
        self.loss_dict = {}
        self.last_time = time.time()

    def update(self, loss_dict):
        if not loss_dict:
            self.loss_dict = loss_dict
        else:
            for key, val in loss_dict.items():
                if key not in self.loss_dict:
                    self.loss_dict[key] = 0.0
                self.loss_dict[key] += val
        self.n_tr_steps += 1

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.last_time

    def output(self, step, epoch, learning_rate):
        """Write out statistics to stdout.
        Args:
           step (int): current step
           epoch (int): current epoch
           lr (float): current learning rate
        """

        logger.info("Epoch [{:2}/{:2}], step [{}/{}], lr {:.6f}, {:.2f} s".format(
            epoch, self.epoch_num, step, self.total_training_steps, learning_rate, self.elapsed_time()))

        for key, val in self.loss_dict.items():
            logger.info("  {:10}: {:.4f} ".format(
                key, val.item() / self.n_tr_steps))

        self.last_time = time.time()

    def log_tensorboard(self, writer, learning_rate, global_step, current_loss=None,
                        eval_scores=None, is_training=True, output_dir=None):
        """ save statistics to tensorboard """
        if is_training:
            writer.add_scalar(tag="losses/loss", scalar_value=current_loss, global_step=global_step)
            writer.add_scalar(tag="learning_rate/lr", scalar_value=learning_rate, global_step=global_step)
        else:
            for metric_name, score in eval_scores:
                writer.add_scalar(tag="eval/%s" % metric_name, scalar_value=score, global_step=global_step)
            if output_dir and "oss://" in output_dir:
                if not io.isdir(output_dir):
                    io.makedirs(output_dir)
                for fname in io.listdir("./easytexminer_tensorboard"):
                    local_file = os.path.join("./easytexminer_tensorboard/", fname)
                    oss_file = os.path.join(output_dir, fname)
                    io.upload(local_file, oss_file)