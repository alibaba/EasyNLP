# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 5:13 pm.
# @Author  : JianingWang
# @File    : freeze.py
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)


class FreezeCallback(TrainerCallback):

    def __init__(self, freeze_epochs, freeze_keyword):
        self.freeze_epochs = freeze_epochs
        self.freeze_keyword = freeze_keyword.split(',')
        self.is_freeze = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        if state.epoch < self.freeze_epochs:
            for name, param in model.named_parameters():
                for keyword in self.freeze_keyword:
                    if keyword in name:
                        param.requires_grad = False
                        logger.info(f'layer {name} is frozen.')
            self.is_freeze = True

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        epoch = state.epoch
        if self.is_freeze and epoch > self.freeze_epochs:
            for name, param in model.named_parameters():
                for keyword in self.freeze_keyword:
                    if keyword in name:
                        param.requires_grad = True
            self.is_freeze = False
