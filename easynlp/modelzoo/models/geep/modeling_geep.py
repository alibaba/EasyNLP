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

import numpy as np
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_geep import GEEPConfig
from ..bert.modeling_bert import *

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hfl/chinese-roberta-wwm-ext",
    "bert-small-uncased",
    "bert-base-uncased",
    "bert-large-uncased",
]

class GEEPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def normal_shannon_entropy(self,p, labels_num):
        entropy = torch.distributions.Categorical(probs=p).entropy()
        normal = -np.log(1.0/labels_num)
        return entropy / normal

    def _difficult_samples_idxs(self, idxs, logits,labels_num,threshold):
            # logits: (batch_size, labels_num)
            probs = nn.Softmax(dim=1)(logits)
            entropys = self.normal_shannon_entropy(probs, labels_num)
            rel_diff_idxs = (entropys > threshold).nonzero().view(-1)
            abs_diff_idxs = torch.tensor([idxs[i] for i in rel_diff_idxs], device=logits.device)
            return abs_diff_idxs, rel_diff_idxs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        classifiers=None,
        mode=None,
        exit_num=None,
        num_labels=None,
        threshold=None,
    ):
        hybrid_emb=hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        hidden_states_sub=[]
        if True:#mode=='train'
            self_layer_len=len(self.layer)
            if mode=='train':
                choosed=list(range(1,exit_num+1))
            else:
                choosed=[]
            choosed.append(self_layer_len-1)
            for skip_layer in range(1,self_layer_len):
                if skip_layer not in choosed:
                    continue
                sub_net_index=list(range(0,skip_layer))
                sub_net_index.append(self_layer_len-1)
                hidden_states=hybrid_emb
                if mode=='inference':
                    batch_size = hidden_states.size(0)
                    inference_logits = torch.zeros(batch_size, num_labels, dtype=hidden_states.dtype, device=hidden_states.device)
                    abs_diff_idxs = torch.arange(0, batch_size, dtype=torch.long, device=hidden_states.device)

                for i in sub_net_index:
                    layer_module=self.layer[i]
                    if i==(self_layer_len-1):
                        hidden_states_sub.append(hidden_states)

                    layer_head_mask = head_mask[i] if head_mask is not None else None
                    past_key_value = past_key_values[i] if past_key_values is not None else None

                    if getattr(self.config, "gradient_checkpointing", False) and self.training:
                        if use_cache:
                            logger.warning(
                                "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                                "`use_cache=False`..."
                            )
                            use_cache = False

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, past_key_value, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer_module),
                            hidden_states,
                            attention_mask,
                            layer_head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                        )
                    else:
                        layer_outputs = layer_module(
                            hidden_states,
                            attention_mask,
                            layer_head_mask,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            past_key_value,
                            output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache += (layer_outputs[-1],)
                    if output_attentions:
                        all_self_attentions = all_self_attentions + (layer_outputs[1],)
                        if self.config.add_cross_attention:
                            all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                
                    if mode=='inference':
                        if i<len(classifiers):
                            new_ptr=i
                        else:
                            if i==(self_layer_len-1):
                                new_ptr=-1
                            else:
                                continue
                        logits_this_layer = classifiers[new_ptr](hidden_states).view(-1, num_labels)
                        inference_logits[abs_diff_idxs] = logits_this_layer

                        # filter easy sample
                        abs_diff_idxs, rel_diff_idxs = self._difficult_samples_idxs(abs_diff_idxs, logits_this_layer,num_labels,threshold) 
                        hidden_states = hidden_states[rel_diff_idxs, :, :]
                        
                        if len(abs_diff_idxs) == 0:#confirmed
                            break

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            if mode=='inference':
                hidden_states_sub=inference_logits
        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=hidden_states_sub,
        )

class GEEPModel(BertPreTrainedModel):
    """

    This is the GEEPModel which bahave like BERTModel. The GEEPClassification application will take this model
    as the backbone and equip this model with attributes like classifiers, exit_num, and threshold. See GEEPClassification in appzoo for more details.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = GEEPEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        classifiers=None,
        mode=None,
        exit_num=None,
        num_labels=None,
        threshold=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            classifiers=classifiers,
            mode=mode,
            exit_num=exit_num,
            num_labels=num_labels,
            threshold=threshold,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
