#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from torch.nn import Embedding


class Manifold(object):
    def allocate_lt(self, N, dim, sparse):
        return Embedding(N, dim, sparse=sparse)

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    def init_weights(self, w, scale=1e-4):
        w.weight.data.uniform_(-scale, scale)

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError

    def norm(self, u, **kwargs):
        if isinstance(u, Embedding):
            u = u.weight
        return u.pow(2).sum(dim=-1).sqrt()

    @abstractmethod
    def half_aperture(self, u):
        """
        Compute the half aperture of an entailment cone.
        As in: https://arxiv.org/pdf/1804.01882.pdf
        """
        raise NotImplementedError

    @abstractmethod
    def angle_at_u(self, u, v):
        """
        Compute the angle between the two half lines (0u and uv
        """
        raise NotImplementedError
