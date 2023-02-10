# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
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
""" Facebook BlenderBot model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

class TransformerConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=54944,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        embedding_size=512,
        ffn_size=2048,
        n_encoder_layers=-1,
        n_decoder_layers=-1,
        n_layers=8,
        n_heads=16,
        dropout=0.1,
        activation="gelu",
        variant="xlm",
        learn_positional_embeddings=True,
        n_positions=512,
        truncate=-1,
        text_truncate=512,
        label_truncate=128,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        output_scaling=1.0,
        n_segments=0,
        checkpoint_activations=False,
        beam_min_length=20,
        beam_block_ngram=3,
        tokenizer='bpe',
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_encoder_layers = n_encoder_layers
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = activation
        self.variant = variant
        self.learn_positional_embeddings = learn_positional_embeddings
        self.n_decoder_layers = n_decoder_layers
        self.n_positions = n_positions
        self.truncate = truncate
        self.text_truncate = text_truncate
        self.label_truncate = label_truncate
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.embeddings_scale = embeddings_scale
        self.output_scaling = output_scaling
        self.n_segments = n_segments
        self.checkpoint_activations = checkpoint_activations
        self.beam_min_length = beam_min_length
        self.beam_block_ngram = beam_block_ngram
        self.tokenizer = tokenizer
