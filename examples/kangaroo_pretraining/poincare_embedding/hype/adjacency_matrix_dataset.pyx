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
from libcpp.unordered_set cimport unordered_set
from libc.math cimport pow
from libc.stdlib cimport RAND_MAX
import threading
import queue

# Thread safe random number generation.  libcpp doesn't expose rand_r...
cdef unsigned long rand_r(unsigned long* seed) nogil:
    seed[0] = (seed[0] * 1103515245) % <unsigned long>pow(2, 32) + 12345
    return seed[0] % RAND_MAX

cdef class AdjacencyDataset:
    cdef public bool burnin
    cdef public int N, qsize, qsamples, qmisses
    cdef public npc.ndarray objects, counts
    cdef public double neg_multiplier, _sample_dampening

    cdef int nnegs, max_tries, batch_size, num_workers, join_count
    cdef long current
    cdef object queue
    cdef list threads

    cdef double [:] S, weights
    cdef long [:] A, ids, neighbors, offsets, perm

    def __cinit__(self, adj, nnegs, batch_size, num_workers, burnin = False,
            sample_dampening=0.75):
        self.burnin = burnin
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_tries = 10 * nnegs
        self.neg_multiplier = 1
        self.queue = queue.Queue(maxsize=num_workers)
        self.nnegs = nnegs
        self._sample_dampening = sample_dampening

        self.ids = adj['ids']
        self.neighbors = adj['neighbors']
        self.offsets = adj['offsets']
        self.weights = adj['weights']
        self.objects = adj['objects']
        self.N = len(self.objects)
        self._setup_alias_tables()

    cdef _setup_alias_tables(self):
        # Setup the necessary data structures for "Alias Method"
        # See Lua Torch impl: https://github.com/torch/torch7/blob/master/lib/TH/generic/THTensorRandom.c
        # Alias method: https://en.wikipedia.org/wiki/Alias_method

        cdef long j, k, i, start, end
        cdef set Tl, Th
        cdef npc.ndarray[npc.long_t, ndim=1] A
        cdef npc.ndarray[npc.double_t, ndim=1] S

        self.counts = np.bincount(self.neighbors, weights=self.weights, minlength=self.N)
        self.counts = self.counts ** self._sample_dampening

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

    def iter(self):
        return self.__iter__()

    def __iter__(self):
        self.perm = np.random.permutation(self.neighbors.shape[0])
        self.qsize = self.qsamples = self.current = self.join_count = self.qmisses = 0
        self.threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(i,))
            t.start()
            self.threads.append(t)
        return self

    def _worker(self, tid):
        cdef long [:,:] memview
        cdef int count
        cdef double [:] weights
        cdef unsigned long seed

        seed = tid
        while self.current < self.neighbors.shape[0]:
            start = self.current
            self.current += self.batch_size

            batch = torch.LongTensor(self.batch_size, self.nnegatives() + 2)
            memview = batch.numpy()
            with nogil:
                count = self._getbatch(start, memview, &seed)
            if count < self.batch_size:
                batch = batch.narrow(0, 0, count)
            self.queue.put((batch, torch.zeros(count).long()))
        self.queue.put(tid)

    def __len__(self):
        return int(np.ceil(float(self.neighbors.shape[0]) / self.batch_size))

    def avg_queue_size(self):
        return float(self.qsize) / self.qsamples

    def queue_misses(self):
        return self.qmisses

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
        if size == 0 and self.join_count == len(self.threads):
            # No more items in queue and we've joined with all worker threads
            raise StopIteration

        item = self.queue.get()
        if isinstance(item, int):
            self.join_count += 1
            self.threads[item].join()  # Thread `item` is finished, join with it...
            return self.next()  # try again...
        self.qsize += size
        self.qsamples += 1
        prevmisses = self.qmisses
        self.qmisses += 1 if size == 0 else 0
        if self.qmisses == 20 and prevmisses == 19:
            print('Warning: not enough threads to keep up with training loop!')
        return item

    cdef long random_node(self, unsigned long* seed) nogil:
        cdef long fu, n
        cdef double u

        if self.burnin:
            u = <double>rand_r(seed) / <double>RAND_MAX * self.N
            fu = <long>u
            if self.S[fu] <= u - fu:
                return self.A[fu]
            else:
                return fu
        else:
            return <long>(<double>rand_r(seed) / <double>RAND_MAX * self.N)

    cdef long binary_search(self, long target, long[:] arr, long l, long r, bool approx) nogil:
        '''
        Binary search.  If the `approx` flag is `True`, then we find the position
        in the array that `target` belongs.  If False, then we return `-1` if
        `target` does not exist
        '''
        cdef long mid, N
        N = r
        while l <= r:
            mid = <long>((l + r) / 2)
            if (approx and arr[mid] <= target and (mid+1 > N or arr[mid+1] > target)) \
                    or arr[mid] == target:
                return mid
            if arr[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return 0 if approx else -1

    cdef long _getbatch(self, long idx, long[:,:] batch, unsigned long* seed) nogil:
        cdef long i, nnodes, ixptr, t, h, l, r, nodeidx, ntries, neighbor_idx
        cdef long rnodeidx, rand_node
        cdef unordered_set[long] negs
        nnodes = self.ids.shape[0]
        i = 0

        while idx < len(self.neighbors) and i < self.batch_size:
            ntries = 0
            neighbor_idx = self.perm[idx]
            nodeidx = self.binary_search(neighbor_idx, self.offsets, 0, nnodes-1, True)

            # nodes for positive sample
            t = self.ids[nodeidx]
            h = self.neighbors[neighbor_idx]

            # left and right boundaries for this node's neighbors
            l = self.offsets[nodeidx]
            r = self.offsets[nodeidx + 1] - 1 if nodeidx + 1 < nnodes else len(self.neighbors) - 1

            batch[i, 0] = t
            batch[i, 1] = h
            ixptr = 2
            negs = unordered_set[long]()

            while ntries < self.max_tries and ixptr < self._nnegatives() + 2:
                rand_node = self.random_node(seed)
                rnodeidx = self.binary_search(rand_node, self.neighbors, l, r, False)
                if rand_node != t and (rnodeidx == -1 or self.weights[rnodeidx] < self.weights[neighbor_idx]):
                    if negs.find(rand_node) == negs.end():
                        batch[i, ixptr] = rand_node
                        ixptr = ixptr + 1
                        negs.insert(rand_node)
                ntries = ntries + 1

            if ixptr == 2:
                batch[i, ixptr] = t
                ixptr += 1

            while ixptr < self._nnegatives() + 2:
                batch[i, ixptr] = batch[i, 2 + <long>(<double>rand_r(seed)/RAND_MAX*(ixptr-2))]
                ixptr = ixptr + 1

            idx = idx + 1
            i = i + 1
        return i

    def nnegatives(self):
        return self._nnegatives()

    cdef int _nnegatives(self) nogil:
        if self.burnin:
            return int(self.neg_multiplier * self.nnegs)
        else:
            return self.nnegs
