# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and The HuggingFace Inc. team.
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

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn

from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_utils import (
    PreTrainedModel,)

class TextCNNEncoder(PreTrainedModel):
    r"""
    This is the abstract class to of cnn encoders


    Args:
        config (:obj: TextCNNConfig):
            The configuration of the TextCNN encoder.
    Examples::

        >>> from easynlp.modelzoo.models.cnn import TextCNNConfig, TextCNNEncoder

        >>> # Initializing a cnn configuration
        >>> configuration = TextCNNConfig()

        >>> # Initializing a model from the cnn-en style configuration
        >>> model = TextCNNEncoder(configuration)
    """

    def __init__(self, config):
        super(TextCNNEncoder, self).__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        embed_size = config.embed_size
        conv_dim = config.conv_dim
        max_seq_len = config.sequence_length
        #kernel_sizes = [int(num) for num in config.kernel_sizes.split(',')]
        kernel_sizes = config.kernel_sizes
        linear_hidden_size = config.hidden_size
        self.cnn_encoder = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=embed_size,
                      out_channels=conv_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=conv_dim,
                      out_channels=conv_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(max_seq_len - kernel_size * 2 + 2))
        ) for kernel_size in kernel_sizes])

        self.fc_layers = nn.Sequential(
            nn.Linear(len(kernel_sizes) * conv_dim, linear_hidden_size),
            nn.BatchNorm1d(linear_hidden_size),
            nn.ReLU(inplace=True),
        )
        #print(self.cnn_encoder)
        #print(self.fc_layers)

    def forward(self, input_ids, **kwargs):
        fact_embeds = self.embedding(input_ids)
        conv_out = [fact_conv(fact_embeds.permute(0, 2, 1)) for fact_conv in self.cnn_encoder]
        conv_out = torch.cat(conv_out, dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        output = self.fc_layers((reshaped))
        return BaseModelOutputWithPooling(pooler_output=output)

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if getattr(module, "weight", False) is not False:
            module.weight.data.normal_(
                mean=0.0, std=0.02)
        if getattr(module, "bias", False) is not False:
            module.bias.data.zero_()


