#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from numpy.random import randint
from . import graph
from .graph_dataset import BatchedDataset

model_name = '%s_dim%d'


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(graph.Dataset):
    def __getitem__(self, i):
        t, h = self.idx[i]
        negs = set()
        ntries = 0
        nnegs = int(self.nnegatives())
        if t not in self._weights:
            negs.add(t)
        else:
            while ntries < self.max_tries and len(negs) < nnegs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    n = int(self.unigram_table[n])
                else:
                    n = randint(0, len(self.objects))
                if (n not in self._weights[t]) or \
                        (self._weights[t][n] < self._weights[t][h]):
                    negs.add(n)
                ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()
