# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cimport numpy as npc
cimport cython

import numpy as np
import torch
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libc.math cimport pow
from libc.stdlib cimport rand, RAND_MAX
import threading
import queue

# Thread safe random number generation.  libcpp doesn't expose rand_r...
cdef unsigned long rand_r(unsigned long* seed) nogil:
    seed[0] = (seed[0] * 1103515245) % <unsigned long>pow(2, 32) + 12345
    return seed[0] % RAND_MAX

cdef class BatchedDataset:
    cdef public list objects
    cdef public bool burnin
    cdef public double neg_multiplier
    cdef public npc.ndarray counts

    cdef long [:, :] idx
    cdef int nnegs, max_tries, N, batch_size, current, num_workers
    cdef double sample_dampening
    cdef vector[unordered_map[long, double]] _weights
    cdef double [:] S
    cdef long [:] A, perm
    cdef object queue
    cdef list threads

    def __cinit__(self, idx, objects, weights, nnegs, batch_size, num_workers,
                  burnin=False, sample_dampening=0.75):
        '''
        Create a dataset for training Hyperbolic embeddings.  Rather than
        allocating many tensors for individual dataset items, we instead
        produce a single batch in each iteration.  This allows us to do a single
        Tensor allocation for the entire batch and filling it out in place.

        Args:
            idx (ndarray[ndims=2]):  Indexes of objects corresponding to co-occurrence.
                I.E. if `idx[0, :] == [4, 19]`, then item 4 co-occurs with item 19
            weights (ndarray[ndims=1]): Weights for each co-occurence.  Corresponds
                to the number of times a pair co-occurred.  (Equal length to `idx`)
            nnegs (int): Number of negative samples to produce with each positive
            objects (list[str]): Mapping from integer ID to hashtag string
            nnegs (int): Number of negatives to produce with each positive
            batch_size (int): Size of each minibatch
            num_workers (int): Number of threads to use to produce each batch
            burnin (bool): ???
        '''
        self.idx = idx
        self.objects = objects
        self.nnegs = nnegs
        self.burnin = burnin
        self.N = len(objects)
        self.counts = np.zeros((self.N), dtype=np.double)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.sample_dampening = sample_dampening
        self._mk_weights(idx, weights)
        self.max_tries = 10 * nnegs
        self.neg_multiplier = 1
        self.queue = queue.Queue(maxsize=num_workers)

    # Setup the weights datastructure and sampling tables
    def _mk_weights(self, npc.ndarray[npc.long_t, ndim=2] idx, npc.ndarray[npc.double_t, ndim=1] weights):
        cdef int i
        cdef long t, h
        cdef set Tl, Th
        cdef npc.ndarray[npc.long_t, ndim=1] A
        cdef npc.ndarray[npc.double_t, ndim=1] S

        self._weights.resize(self.N)

        for i in range(idx.shape[0]):
            t = idx[i, 0]
            h = idx[i, 1]
            self.counts[h] += weights[i]
            self._weights[t][h] = weights[i]

        self.counts = self.counts ** self.sample_dampening

        if self.burnin:
            # Setup the necessary data structures for "Alias Method"
            # See Lua Torch impl: https://github.com/torch/torch7/blob/master/lib/TH/generic/THTensorRandom.c
            # Alias method: https://en.wikipedia.org/wiki/Alias_method
            S = (self.counts / np.sum(self.counts)) * self.counts.shape[0]
            A = np.arange(0, self.counts.shape[0], dtype=np.long)
            Tl = set(list((S < 1).nonzero()[0]))
            Th = set(list((S > 1).nonzero()[0]))

            while len(Tl) > 0 and len(Th) > 0:
                j = Tl.pop()
                k = Th.pop()
                S[k] = S[k] - 1 + S[j]
                A[j] = k
                if S[k] < 1:
                    Tl.add(k)
                elif S[k] > 1:
                    Th.add(k)
            self.S = S
            self.A = A

    def __iter__(self):
        self.perm = np.random.permutation(len(self.idx))
        self.current = 0
        self.threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(i,))
            t.start()
            self.threads.append(t)
        return self

    cpdef _worker(self, i):
        cdef long [:,:] memview
        cdef long count

        while self.current < self.idx.shape[0]:
            current = self.current
            self.current += self.batch_size
            ix = torch.LongTensor(self.batch_size, self.nnegatives() + 2)
            memview = ix.numpy()
            with nogil:
                count = self._getbatch(current, memview)
            if count < self.batch_size:
                ix = ix.narrow(0, 0, count)
            self.queue.put((ix, torch.zeros(ix.size(0)).long()))
        self.queue.put(i)

    def iter(self):
        return self.__iter__()

    def __len__(self):
        return int(np.ceil(float(self.idx.shape[0]) / self.batch_size))

    def __next__(self):
        return self.next()

    def next(self):
        '''
        Python visible function for indexing the dataset.  This first
        allocates a tensor, and then modifies it in place with `_getitem`

        Args:
            idx (int): index into the dataset
        '''
        size = self.queue.qsize()
        if size == 0 and all([not(t.is_alive()) for t in self.threads]):
            # No more items in queue and we've joined with all worker threads
            raise StopIteration
        item = self.queue.get()
        if isinstance(item, int):
            self.threads[item].join()  # Thread `item` is finished, join with it...
            return self.next()  # try again...
        return item

    cdef public long _getbatch(self, int i, long[:,:] ix) nogil:
        '''
        Fast internal C method for indexing the dataset/negative sampling

        Args:
            i (int): Index into the dataset
            ix (long [:]) - A C memoryview of the result tensor that we will
                return to Python
            N (int): Total number of unique objects in the dataset (convert to raw C)
        '''
        cdef long t, h, n, fu
        cdef int ntries, ixptr, idx, j
        cdef unordered_set[long] negs
        cdef double weight_th, u
        cdef unsigned long seed

        seed = i
        j = 0

        while j < self.batch_size and i + j < self.perm.shape[0]:
            ntries = 0

            idx = self.perm[i + j]
            t = self.idx[idx, 0]
            h = self.idx[idx, 1]

            ix[j, 0] = t
            ix[j, 1] = h
            ixptr = 2

            weight_th = self._weights[t][h]

            negs = unordered_set[long]()

            while ntries < self.max_tries and negs.size() < self._nnegatives():
                if self.burnin:
                    u = <double>rand_r(&seed) / <double>RAND_MAX * self.N
                    fu = <int>u
                    if self.S[fu] <= u - fu:
                        n = self.A[fu]
                    else:
                        n = fu
                else:
                    n = <long>(<double>rand_r(&seed) / <double>RAND_MAX * self.N)
                if n != t and (self._weights[t].find(n) == self._weights[t].end() or (self._weights[t][n] < weight_th)):
                    if negs.find(n) == negs.end():
                        ix[j, ixptr] = n
                        ixptr = ixptr + 1
                        negs.insert(n)
                ntries = ntries + 1

            if negs.size() == 0:
                ix[j, ixptr] = t
                ixptr = ixptr + 1

            while ixptr < self._nnegatives() + 2:
                ix[j, ixptr] = ix[j, 2 + <long>(<double>rand_r(&seed)/RAND_MAX*(ixptr-2))]
                ixptr = ixptr + 1
            j = j + 1
        return j

    def nnegatives(self):
        return self._nnegatives()

    cdef int _nnegatives(self) nogil:
        if self.burnin:
            return int(self.neg_multiplier * self.nnegs)
        else:
            return self.nnegs
