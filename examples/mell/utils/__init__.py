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
import random
import re

import numpy as np
import torch
import torch.nn as nn

from .distribution_utils import init_gpu_env, distributed_call_main
from .io_utils import IO, DefaultIO, OSSIO, parse_oss_buckets
from .logger import init_logger


class _IOWrapper:
    def __init__(self):
        self._io = DefaultIO()

    def set_io(self, new_io):
        self._io = new_io

    def __getattr__(self, name):
        if hasattr(self._io, name):
            return getattr(self._io, name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            raise AttributeError(f"'io' object has no attribute '{name}'")

    def __str__(self):
        return self._io.__name__


io = _IOWrapper()


def copy_weights_for_same_module(copied_module, copying_module):
    if isinstance(copied_module, nn.Parameter):
        copying_module.data.copy_(copied_module.data)
        return
    tp = copied_module.named_parameters()
    for name, weights in copying_module.named_parameters():
        _, weights_ = next(tp)
        weights.data.copy_(weights_.data)


def get_dir_name(file_path):
    if io.isdir(file_path):
        return file_path
    else:
        return os.path.dirname(file_path)


def mapping_teacher_student_layers(teacher_layers, student_layers, mapping_strategy="skip"):
    new_teacher_layers = list()
    if mapping_strategy == "first":
        for i in range(len(student_layers)):
            new_teacher_layers.append(teacher_layers[i])
    elif mapping_strategy == "last":
        for i in reversed(range(len(student_layers))):
            t_layer = len(teacher_layers) - i - 1
            new_teacher_layers.append(teacher_layers[t_layer])
    elif mapping_strategy == "skip":
        skip_interval = len(teacher_layers) // len(student_layers)
        for i in range(len(student_layers)):
            new_teacher_layers.append(teacher_layers[i * skip_interval + skip_interval - 1])
    elif mapping_strategy == "only_last":
        new_teacher_layers.append(teacher_layers[-1])
        student_layers = [student_layers[-1]]
    else:
        raise NotImplementedError
    return new_teacher_layers, student_layers


def parse_student_config(stu_config_str):
    stu_config_json = dict()
    for item in re.findall("\w+=[\w/.\-:\/]+", stu_config_str):
        key, val = item.strip().split("=")
        stu_config_json[key] = val
    assert "model_type" in stu_config_json
    if stu_config_json["model_type"] == "bert":
        from ..layers.transformers import BertConfig
        stu_config_json = BertConfig.from_json_file(stu_config_json).to_dict()
    else:
        raise NotImplementedError("%s model not implemented!" % stu_config_json["model_type"])
    return stu_config_json


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_running_envs(process_id, cfg):
    init_gpu_env(process_id, cfg)
    init_logger(local_rank=cfg.global_rank)
    set_random_seed(cfg.seed)

    if cfg.buckets is not None:
        access_key_id, access_key_secret, hosts, buckets = parse_oss_buckets(cfg.buckets)
        new_io = OSSIO(access_key_id=access_key_id,
                   access_key_secret=access_key_secret,
                   hosts=hosts,
                   buckets=buckets)
        io.set_io(new_io)

    if "odps://" in cfg.tables:
        cfg.read_odps = True
    else:
        cfg.read_odps = False

