#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply
