#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import numpy as np
import timeit
import gc
from tqdm import tqdm
from torch.utils import data as torch_data

_lr_multiplier = 0.1


def train(
        device,
        model,
        data,
        optimizer,
        opt,
        log,
        rank=1,
        queue=None,
        ctrl=None,
        checkpointer=None,
        progress=False
):
    if isinstance(data, torch_data.Dataset):
        loader = torch_data.DataLoader(data, batch_size=opt.batchsize,
            shuffle=True, num_workers=opt.ndproc)
    else:
        loader = data

    epoch_loss = th.Tensor(len(loader))
    counts = th.zeros(model.nobjects, 1).to(device)
    for epoch in range(opt.epoch_start, opt.epochs):
        epoch_loss.fill_(0)
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            if rank == 1:
                log.info(f'Burn in negs={data.nnegatives()}, lr={lr}')

        loader_iter = tqdm(loader) if progress and rank == 1 else loader
        for i_batch, (inputs, targets) in enumerate(loader_iter):
            elapsed = timeit.default_timer() - t_start
            inputs = inputs.to(device)
            targets = targets.to(device)

            # count occurrences of objects in batch
            if hasattr(opt, 'asgd') and opt.asgd:
                counts = th.bincount(inputs.view(-1), minlength=model.nobjects)
                counts = counts.clamp_(min=1)
                getattr(counts, 'floor_divide_', counts.div_)(inputs.size(0))
                counts = counts.double().unsqueeze(-1)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr, counts=counts)
            epoch_loss[i_batch] = loss.cpu().item()
        if rank == 1:
            if hasattr(data, 'avg_queue_size'):
                qsize = data.avg_queue_size()
                misses = data.queue_misses()
                log.info(f'Average qsize for epoch was {qsize}, num_misses={misses}')

            if queue is not None:
                queue.put((epoch, elapsed, th.mean(epoch_loss).item(), model))
            elif ctrl is not None and epoch % opt.eval_each == (opt.eval_each - 1):
                with th.no_grad():
                    ctrl(model, epoch, elapsed, th.mean(epoch_loss).item())
            else:
                log.info(
                    'json_stats: {'
                    f'"epoch": {epoch}, '
                    f'"elapsed": {elapsed}, '
                    f'"loss": {th.mean(epoch_loss).item()}, '
                    '}'
                )
            if checkpointer and hasattr(ctrl, 'checkpoint') and ctrl.checkpoint:
                checkpointer(model, epoch, epoch_loss)

        gc.collect()
