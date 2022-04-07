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
from torch import nn


def mse_loss(inputs, targets, **kwargs):
    """MSE loss.

    Args:
        inputs: input tensor
        targets: prediction tensor
    """
    return torch.nn.functional.mse_loss(inputs, targets, **kwargs)


def cross_entropy(input,
                  target,
                  weight=None,
                  size_average=None,
                  ignore_index=-100,
                  reduce=None,
                  reduction='mean'):
    """Cross Entropy loss.

    Args:
        input: input tensor
        target: prediction tensor
        weight: weighted cross-entropy loss (sample level weights)
        size_average: size average
        ignore_index: ignore index
        reduction: default 'mean' reduction
    """
    return F.cross_entropy(input, target, weight, size_average, ignore_index,
                           reduce, reduction)


def vanilla_loss(s_logits,
                 t_logits,
                 labels,
                 alpha=0.2,
                 temperature=1,
                 **kwargs):
    """Vanilla KD loss.

    Args:
        s_logits: student logits
        t_logits: target logits
        alpha: kd loss weight
        temperature: temperature
    """
    T = temperature
    kd_loss = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                       F.softmax(t_logits / T, dim=1),
                       reduction='batchmean') * T * T
    # nll_loss = F.cross_entropy(s_logits, labels, reduction='mean')
    nll_loss = cross_entropy(s_logits, labels)

    return alpha * kd_loss + (1 - alpha) * nll_loss


def multi_label_sigmoid_cross_entropy(input,
                                      target,
                                      weight=None,
                                      size_average=None,
                                      ignore_index=-100,
                                      reduce=None,
                                      reduction='mean'):
    """MultiLabel Sigmoid Cross Entropy loss.

    Args:
        input: input tensor
        target: prediction tensor
        weight: weighted cross-entropy loss (sample level weights)
        size_average: size average
        ignore_index: ignore index
        reduction: default 'mean' reduction
    """
    loss = nn.BCEWithLogitsLoss()
    return loss(input, target)


def soft_input_cross_entropy(input,
                             target,
                             weight=None,
                             size_average=None,
                             ignore_index=-100,
                             reduce=None,
                             reduction='mean'):
    """Soft Input Cross Entropy loss.

    Args:
        input: input tensor
        target: prediction tensor
        weight: weighted cross-entropy loss (sample level weights)
        size_average: size average
        ignore_index: ignore index
        reduction: default 'mean' reduction
    """
    input = torch.log(input)
    return F.nll_loss(input, target, weight, size_average, ignore_index,
                      reduce, reduction)


def matching_embedding_hinge_loss(emb1, emb2, margin=0.3):
    """Hinge loss for embeddings.

    Args:
        emb1: embedding tensor
        emb2: embedding tensor
        margin: margin (default 0.3)
    """
    return F.hinge_embedding_loss(emb1, emb2, margin)


def matching_embedding_circle_loss(emb1, emb2, margin=0.45, gamma=32):
    raise NotImplementedError


def soft_cross_entropy(input, targets):
    student_likelihood = torch.nn.functional.log_softmax(input, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (-targets_prob * student_likelihood).sum(dim=-1).mean()


def cosine_embedding_loss(input1, input2, target, **kwargs):
    return torch.nn.functional.cosine_embedding_loss(input1,
                                                     input2,
                                                     target,
                                                     reduction='mean',
                                                     **kwargs)
