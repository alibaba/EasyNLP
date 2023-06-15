# coding=utf-8
# Copyright 2022, Google and HuggingFace Inc.
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
""" Switch Transformers model configuration"""
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/switch-base-8": "https://huggingface.co/google/switch-base-8/blob/main/config.json",
}


class MTAConfig(PretrainedConfig):

    def __init__(
        self,
        vocab_size=32128,
        d_model=768,
        d_kv=64,
        d_ff=2048,
        expert_capacity=64,
        num_layers=6,
        num_sparse_encoder_layers=0,
        num_decoder_layers=6,
        num_sparse_decoder_layers=0,
        num_heads=6,
        num_experts=8,
        router_type="tokens_masked",
        my_hidden_size=768,
        router_bias=False,
        router_jitter_noise=0.01,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        add_router_probs=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff

        self.num_sparse_encoder_layers = num_sparse_encoder_layers

        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_sparse_decoder_layers = num_sparse_decoder_layers

        # This tells us, each how many encoder layer we'll have to set a sparse layer.
        if self.num_sparse_encoder_layers > 0:
            self.encoder_sparse_step = self.num_layers // self.num_sparse_encoder_layers
        else:
            self.encoder_sparse_step = self.num_layers  # HACK: this will create 0 sparse layers

        # This tells us, each how many encoder layer we'll have to set a sparse layer.
        if self.num_sparse_decoder_layers > 0:
            self.decoder_sparse_step = self.num_decoder_layers // self.num_sparse_decoder_layers
        else:
            self.decoder_sparse_step = self.num_decoder_layers  # HACK: this will create 0 sparse layers

        self.num_heads = num_heads
        self.router_type = router_type
        self.my_hidden_size = my_hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.router_bias = router_bias
        self.router_jitter_noise = router_jitter_noise
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype

        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.add_router_probs = add_router_probs

        self.router_z_loss_coef = router_z_loss_coef
        self.router_aux_loss_coef = router_aux_loss_coef

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
