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

import json
import os
import warnings
import torch
import torch.nn as nn

from .albert import AlbertConfig
from .bert import BertLayerNorm, BertConfig
from .gpt2 import GPT2Config
from utils import adapter, io, get_dir_name
from utils.logger import logger


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


def get_model_type_from_pretrained(pretrained_model_name_or_path):
    config_file = os.path.join(get_dir_name(pretrained_model_name_or_path), CONFIG_NAME)
    with io.open(config_file) as f:
        config_json = json.load(f)

    if "model_type" not in config_json:
        warnings.warn("`model_type` not found in %s, set it to `bert` by default." % config_file)
        model_type = "bert"
    else:
        model_type = config_json["model_type"]
    return model_type


class BaseModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(BaseModel, self).__init__()
        self.config = config
        self.extra_model_params = dict()

    @property
    def arch(self):
        if hasattr(self, "config"):
            config = self.config
        elif hasattr(self, "bert") and hasattr(self.bert, "config"):
            config = self.bert.config
        else:
            config = None

        if config:
            if isinstance(config, str):
                return config
            elif isinstance(config, dict):
                tmp = {key: val for key, val in config.items()}
                tmp["extra_model_params"] = self.extra_model_params
                return json.dumps(tmp, indent=4)
            else:
                tmp = {key: val for key, val in config.__dict__.items()}
                tmp["extra_model_params"] = self.extra_model_params
                return json.dumps(tmp, indent=4)
        else:
            return self.__str__()

    def init_model_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config_cls=None, adapter_fn=None, *args, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        config_dict = kwargs.get('config_dict', None)
        kwargs.pop('config_dict', None)

        if config_cls is None:
            if config_dict:
                model_type = config_dict.get("model_type", "bert")
            else:
                model_type = get_model_type_from_pretrained(pretrained_model_name_or_path)
            if model_type in ["bert", "roberta"]:
                config_cls = BertConfig
            elif model_type == "albert":
                config_cls = AlbertConfig
            elif model_type == "gpt2":
                config_cls = GPT2Config
            else:
                raise NotImplementedError
        if config_dict:
            config = config_cls.from_dict(config_dict)
        else:
            config = config_cls.from_json_file(
                os.path.join(get_dir_name(pretrained_model_name_or_path), CONFIG_NAME))

        # Instantiate model.
        model = cls(config, *args, **kwargs)

        # Check if the model is from tensorflow checkpoint
        is_tf_checkpoint = False
        if io.exists(pretrained_model_name_or_path + ".index") or \
                io.exists(pretrained_model_name_or_path + ".meta"):
            is_tf_checkpoint = True

        if is_tf_checkpoint:
            if adapter_fn:
                adapter_fn(model, pretrained_model_name_or_path)
            else:
                adapter.load_bert_tf_checkpoint_weights(model, pretrained_model_name_or_path)

        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            if not io.exists(weights_path):
                return model
            logger.info("Loading model {}".format(weights_path))
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

        if config.model_type == "gpt2":
            new_state_dict = {"gpt2." + key.replace("transformer.", ""): val for key, val in state_dict.items()}
            state_dict = new_state_dict

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'

        logger.info('Loading model...')
        load(model, prefix=start_prefix)
        logger.info('Load finished!')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))

        return model