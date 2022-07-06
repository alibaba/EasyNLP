# coding=utf-8
# Copyright 2020 Alibaba PAI team, The Google AI Language Team Authors, and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
""" ARTIST model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

class ARTISTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`MinGPTModel`. 
    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the GEEP model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`GEEPModel`.
    """
    model_type = "artist"

    def __init__(
        self,
        vocab_size=37512,
        img_vocab_size=16384,
        text_vocab_size=21128,
        block_size=288,
        n_layer=12,
        n_head=12,
        n_embd=768,
        embd_pdrop = 0.,
        resid_pdrop = 0.,
        attn_pdrop = 0.,
        n_unmasked=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.img_vocab_size = img_vocab_size
        self.text_vocab_size = text_vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_unmasked = n_unmasked
        for k,v in kwargs.items():
            setattr(self, k, v)
