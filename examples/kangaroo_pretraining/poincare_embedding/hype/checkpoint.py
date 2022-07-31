#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import torch
import warnings
from hype.path_manager import path_manager


def upgrade_state_dict(state):
    '''
    Used to upgrade old checkpoints to deal with breaking changes
    '''
    conf = state['conf']

    # Previously we only had `-manifold`.  if this is an old checkpoint, then
    # update the `conf` to use the same manifold and "distance" model...
    if 'model' not in conf:
        warnings.warn(
            'Missing `model` field in checkpoint config.'
            '  Assuming `distance`.'
        )
        conf['model'] = 'distance'
    return state


class LocalCheckpoint(object):
    def __init__(self, path, include_in_all=None, start_fresh=False):
        self.path = path
        self.start_fresh = start_fresh
        self.include_in_all = {} if include_in_all is None else include_in_all

    def initialize(self, params):
        if not self.start_fresh and os.path.isfile(self.path):
            print(f'Loading checkpoint from {self.path}')
            with path_manager.open(self.path, 'rb') as fin:
                return torch.load(fin)
        else:
            return params

    def save(self, params, tries=10):
        try:
            with path_manager.open(self.path, 'wb') as fout:
                torch.save({**self.include_in_all, **params}, fout)
        except Exception as err:
            if tries > 0:
                print(f'Exception while saving ({err})\nRetrying ({tries})')
                time.sleep(60)
                self.save(params, tries=(tries - 1))
            else:
                print("Giving up on saving...")
