# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 9:17 pm.
# @Author  : JianingWang
# @File    : LoggerCallback.py
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

from config import TrainingArguments

logger = logging.getLogger(__name__)


class LoggerCallback(TrainerCallback):

    def __init__(self):
        for handler in logger.parent.handlers:
            if type(handler) == logging.FileHandler:
                logger.addHandler(handler)
        logger.propagate = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if state.is_world_process_zero:
            logger.info(str(logs))

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass