# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 7:33 pm.
# @Author  : JianingWang
# @File    : ema.py
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from copy import deepcopy


class ExponentialMovingAveragingCallback(TrainerCallback):

    def __init__(self, decay):
        self.decay = decay
        self.average_model = None
        self.model_weights = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.average_model = deepcopy(model)
        self.model_weights = deepcopy(model)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.update_parameters(model)
        if control.should_evaluate:
            self.transfer_weights(model, self.model_weights)
            self.transfer_weights(self.average_model, model)
        if control.should_training_stop:
            self.transfer_weights(self.average_model, model)


    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        if not control.should_save:
            self.transfer_weights(self.model_weights, model)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.transfer_weights(self.model_weights, model)

    @staticmethod
    def transfer_weights(src_model, dst_model):
        for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def update_parameters(self, model):
        for p_ema, p_model in zip(self.average_model.parameters(), model.parameters()):
            device = p_ema.device
            p_ema_ = p_ema.detach()
            p_model_ = p_model.detach().to(device)
            src = (1.0 - self.decay) * p_model_ + self.decay * p_ema_
            p_ema_.copy_(src)
