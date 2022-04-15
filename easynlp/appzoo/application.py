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
import torch
import torch.nn as nn

from ..utils import io
from ..utils.logger import logger
from ..modelzoo import AutoConfig
from ..modelzoo.modeling_utils import print_init_keys_info


class Application(nn.Module):

    def __init__(self):
        super().__init__()
    
    def init_weights(self):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Instantiate model.
        if "modelzoo" in pretrained_model_name_or_path:
            return cls(pretrained_model_name_or_path)

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(from_config=config)
        state_dict = None
        
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if not io.exists(weights_path):
            print_init_keys_info()
            return model
        with io.open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys,
                                         unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        logger.info('Loading model...')
        load(model, prefix=start_prefix)
        logger.info('Load finished!')

        expected_keys = list(model.state_dict().keys())
        loaded_keys = list(state_dict.keys())
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        print_init_keys_info(loaded_keys, expected_keys, unexpected_keys, missing_keys, error_msgs)
        
        return model
