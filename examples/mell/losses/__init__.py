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

import torch
import torch.nn.functional as F
from .kd_loss import (
    soft_cross_entropy,
    soft_cross_entropy_tinybert,
    soft_kl_div_loss,
    mse_loss,
    soft_input_mse_loss,
    cosine_embedding_loss
)


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction)


def soft_input_cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    input = torch.log(input)
    return F.nll_loss(input, target, weight, size_average, ignore_index, reduce, reduction)