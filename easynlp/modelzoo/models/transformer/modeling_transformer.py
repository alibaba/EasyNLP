from typing import Dict, Type, Optional, Tuple, Union
from abc import ABC

import torch
import torch.cuda
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

from ...modeling_utils import (PreTrainedModel)
from .configuration_transformer import TransformerConfig

from parlai.core.opt import Opt
from parlai.core.params import default
from parlai.core.torch_agent import DictionaryAgent
from parlai.utils.misc import warn_once
from parlai.utils.fsdp import fsdp_wrap
from parlai.nn.checkpoint import checkpoint_wrapper
from parlai.utils.torch import PipelineHelper
from parlai.utils.torch import neginf

LAYER_NORM_EPS = 1e-5  # Epsilon for layer norm.

def create_position_codes(n_pos, dim, out):
    """
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)

def get_n_positions_from_options(opt: Opt):
    """
    Determine n_positions from options dict.
    """
    if opt.get('n_positions'):
        # if the number of positions is explicitly provided, use that
        n_positions = opt['n_positions']
    else:
        # else, use the worst case from truncate
        n_positions = max(
            opt.get('truncate') or 0,
            opt.get('text_truncate') or 0,
            opt.get('label_truncate') or 0,
        )
        if n_positions == 0:
            # default to 1024
            n_positions = 1024
    if n_positions < 0:
        raise ValueError('n_positions must be positive')
    return n_positions

def create_embeddings(dictionary, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size**-0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e

class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    For documentation on parameters that are take directly from opt,
    see parlai/agents/transformer/transformer.py

    :param opt: ParlAI-parsed options.
    :param vocabulary_size: Count of tokens/words in the dictionary.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param str reduction_type: Type of reduction at the end of the encoder.
    :param int n_positions: Size of the position embeddings matrix.
    :param int n_segments: Number of segments/lang/sentence embeddings.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    """

    def __init__(
        self,
        opt: Opt,
        vocabulary_size: int,
        embedding: Optional[nn.Embedding] = None,
        padding_idx: int = 0,
        reduction_type: str = 'mean',
        n_positions: Optional[int] = None,
        n_segments: Optional[int] = None,
        embeddings_scale: Optional[bool] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        variant: Optional[str] = None,
        output_scaling: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.opt = opt
        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_encoder_layers']
            if opt.get('n_encoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.embeddings_scale = default(
            embeddings_scale, opt.get('embeddings_scale', False)
        )
        self.reduction_type = reduction_type
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout_frac = default(dropout, opt.get('dropout', 0.0))
        self.dropout = nn.Dropout(p=self.dropout_frac)
        self.activation = default(activation, opt.get('activation', 'relu'))
        self.variant = default(variant, opt.get('variant', 'aiayn'))
        self.n_segments = default(n_segments, opt.get('n_segments', 0))

        self.n_positions = default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        assert (
            self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        # check input formats:
        if embedding is not None:
            assert (
                self.embedding_size is None
                or self.embedding_size == embedding.weight.shape[1]
            ), "Embedding dim must match the embedding size."

        if embedding is not None:
            self.embeddings = embedding
        else:
            raise AssertionError(
                "This code should not execute. Left here in case we want to enable it."
            )
            assert self.padding_idx is not None
            self.embeddings = nn.Embedding(
                vocabulary_size, self.embedding_size, padding_idx=padding_idx
            )
            nn.init.normal_(self.embeddings.weight, 0, self.embedding_size**-0.5)

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size**-0.5
            )

        # embedding normalization
        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        if self.n_segments >= 1:
            self.segment_embeddings = nn.Embedding(self.n_segments, self.dim)
            nn.init.normal_(self.segment_embeddings.weight, 0, self.dim**-0.5)

        # build the model
        self.layers = self.build_layers()
        self.output_scaling = default(output_scaling, opt.get('output_scaling', 1.0))

    def build_layers(self) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = self.swappables.layer(  # type: ignore
                self.opt,
                attention_dropout=self.opt.get('attention_dropout', 0.0),
                relu_dropout=self.opt.get('relu_dropout', 0.0),
                dropout=self.dropout_frac,
                variant=self.variant,
                activation=self.activation,
            )
            if self.opt.get('checkpoint_activations'):
                layer = checkpoint_wrapper(layer)
            layers.append(fsdp_wrap(layer))
        return layers

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.

        :return (tensor, mask):
            return embedded input and mask
        """
        mask = input != self.padding_idx
        if positions is None:
            positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)

        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        position_embs = self.position_embeddings(positions).expand_as(tensor)
        tensor = tensor + position_embs

        if self.n_segments >= 1:
            if segments is None:
                segments = torch.zeros_like(input)  # type: ignore
            tensor = tensor + self.segment_embeddings(segments)

        return tensor, mask

    def forward_layers(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Apply transformer layers to input.

        :param tensor:
            embedded input
        :param mask:
            mask of input

        :return tensor:
            return embedding after applying transformer layers
        """
        if getattr(self.layers, 'is_model_parallel', False):
            # factored out for readability. It is equivalent to the other
            # condition
            tensor = self._apply_model_parallel(tensor, mask)
        else:
            for i in range(self.n_layers):
                tensor = self.layers[i](tensor, mask)

        return tensor

    def reduce_output(
        self, tensor: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        """
        Reduce transformer output at end of forward pass.

        :param tensor:
            encoded input
        :param mask:
            mask for encoded input

        :return (tensor, mask):
            returns the reduced tensor, and mask if appropriate
        """
        tensor *= self.output_scaling
        if self.reduction_type == 'first':
            return tensor[:, 0, :], None
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0], None
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output, None
        elif self.reduction_type is None or 'none' in self.reduction_type:
            return tensor, mask
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen] segments:
            If provided, additionally adds ``segments`` as extra embedding features.
        """
        # embed input
        tensor, mask = self.forward_embedding(input, positions, segments)

        if self.variant == 'xlm' or self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)

        # apply transformer layers
        tensor = self.forward_layers(tensor, mask)

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        # reduce output
        tensor, out_mask = self.reduce_output(tensor, mask)
        if out_mask is not None:
            return tensor, out_mask
        else:
            return tensor

    def _apply_model_parallel(self, tensor, mask):
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, mask))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, s_mask = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor = self.layers[layer_no](s_tensor, s_mask)
            chunks[chunk_idx] = PipelineHelper.chunk_to((s_tensor, s_mask), next_device)

        tensor_out, mask_out = PipelineHelper.join(chunks)
        return tensor_out

class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention; this is the core workhorse of the Transformer.

    See Vaswani (2017) for an extensive description.
    """

    def __init__(
        self, opt: Opt, n_heads: int = None, dim: int = None, dropout: float = 0
    ):
        super(MultiHeadAttention, self).__init__()

        n_heads = default(n_heads, opt['n_heads'])
        dim = default(dim, opt['embedding_size'])

        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(  # type: ignore
        # TODO: remove type ignore with pytorch 1.5:
        # https://github.com/pytorch/pytorch/pull/31057
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        static_kv: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: tensor in which True means that we are allowing attention and False
          means we are blocking it. Mask is:
          - [B, key_len] (encoder self-attn and decoder enc/dec attn)
          - [B, query_len, key_len] (decoder self-attn)
          - [B, 1, key_len] (decoder self-attn with incr_state caching)
        :param incr_state: dictionary with values representing the previous states of
          the key, value, and mask
        :param static_kv: True if the key and value are held constant during decoding
          (as in encoder/decoder attention)
        :return: (
          final attended tensor,
          new incremental state,
          key/value-multiplied tensor before softmax,
        )
        """

        batch_size, query_len, dim = query.size()
        assert (
            dim == self.dim
        ), 'Dimensions do not match: {} query vs {} configured'.format(dim, self.dim)
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
            _, _key_len, dim = query.size()
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        assert key is not None  # let mypy know we sorted this
        _, _key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))

        # Prepend incremental states. For each of the key, value, and mask, see if
        # a previous incremental state exists, and if so, reshape it to match the shape
        # of the new state. Concatenate the previous and new states to match what the
        # full state would have been if we had not cached. (If we are using static_kv,
        # these three states are unchanging, so just re-use the cached states.)
        if incr_state is None:
            incr_state = {}
        if 'prev_key' in incr_state:
            prev_key = incr_state['prev_key'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, prepare_head(self.k_lin(key))], dim=1)
        else:
            k = prepare_head(self.k_lin(key))
        if 'prev_value' in incr_state:
            prev_value = incr_state['prev_value'].view(
                batch_size * n_heads, -1, dim_per_head
            )
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, prepare_head(self.v_lin(value))], dim=1)
        else:
            v = prepare_head(self.v_lin(value))
        if 'prev_mask' in incr_state:
            if static_kv:
                mask = incr_state['prev_mask']
            else:
                # Mask will be of size (B x query_len x key_len)
                # During incremental decoding the query will only represent the next token,
                # whereas the key/value will represent the entire sequence thus far.
                # In such a case, we only want to look at the last element of the mask in the query dimension.
                prev_mask = incr_state['prev_mask'][:, -query_len:, :]
                mask = torch.cat([prev_mask, mask], dim=2)
                # Prepend along the key_len dimension (analogous to incr_state['prev_key'])

        # Save new incremental states. We reshape to allow for reordering along batch
        # dimension.
        new_incr_state = {
            'prev_key': k.view(batch_size, n_heads, -1, dim_per_head),
            'prev_value': v.view(batch_size, n_heads, -1, dim_per_head),
            'prev_mask': mask,
        }

        full_key_len = k.size(1)
        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, full_key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, full_key_len)
            .view(batch_size * n_heads, query_len, full_key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(
            dot_prod, dim=-1, dtype=torch.float  # type: ignore
        ).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out, new_incr_state, dot_prod

    def reorder_incremental_state(
        self, incremental_state: Dict[str, torch.Tensor], inds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Reorder the input incremental-state tensors.
        """
        return {
            key: torch.index_select(val, 0, inds.to(val.device)).contiguous()
            for key, val in incremental_state.items()
        }

class TransformerFFN(nn.Module):
    """
    Implements the FFN part of the transformer.
    """

    def __init__(
        self,
        opt: Opt,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
        **kwargs,
    ):
        super(TransformerFFN, self).__init__(**kwargs)

        dim = default(dim, opt['embedding_size'])
        dim_hidden = default(dim_hidden, opt['ffn_size'])

        self.opt = opt
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = F.gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x

DecoderLayerIncrState = Dict[str, Dict[str, torch.Tensor]]

class BaseTransformerDecoderLayer(nn.Module, ABC):
    """
    Implements functionality common to all transformer decoder layer variants. Subclass
    this if you'd like to modify the behavior of any layer in a transformer decoder.

    While this code is functional, it is not intended to be instantiated directly. If
    this functionality is desired as-is, use TransformerDecoderOnlyLayer instead to gain
    the ability to swap self-attention and feedforward classes at instantiation.
    """

    def __init__(
        self,
        opt: Opt,
        n_heads: int = None,
        embedding_size: int = None,
        ffn_size: int = None,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'relu',
        variant: str = 'aiayn',
        **kwargs,
    ):
        super().__init__()

        n_heads = default(n_heads, opt['n_heads'])
        embedding_size = default(embedding_size, opt['embedding_size'])
        ffn_size = default(ffn_size, opt['ffn_size'])

        self.opt = opt
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.variant = variant
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = self.build_self_attention(
            n_heads=n_heads, dim=embedding_size, dropout=attention_dropout
        )
        self.norm1 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

        self.ffn = self.build_feedforward(
            dim=embedding_size,
            dim_hidden=ffn_size,
            relu_dropout=relu_dropout,
            activation=activation,
        )
        self.norm3 = torch.nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)

    def build_self_attention(
        self, n_heads: int = None, dim: int = None, dropout: float = 0
    ) -> MultiHeadAttention:
        return MultiHeadAttention(
            opt=self.opt, n_heads=n_heads, dim=dim, dropout=dropout
        )

    def build_feedforward(
        self,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ) -> TransformerFFN:
        return TransformerFFN(
            opt=self.opt,
            dim=dim,
            dim_hidden=dim_hidden,
            relu_dropout=relu_dropout,
            activation=activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *extra_args,
        incr_state: Optional[DecoderLayerIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderLayerIncrState]:
        """
        Forward pass.

        The incremental state is a dict with values for self-attention states.
        """
        if incr_state is None:
            incr_state = {}

        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm1(x)

        # don't peak into the future!
        x, final_self_attn_incr_state = self.self_attention(
            query=x,
            mask=decoder_mask,
            incr_state=incr_state.get('self_attn'),
            static_kv=False,
            **kwargs,
        )[:2]
        x = self.dropout(x)  # --dropout
        x = x + residual
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm1(x)

        # finally the ffn
        residual = x
        if self.variant == 'prelayernorm':
            x = self.norm3(x)
        x = self.ffn(x, **kwargs)
        x = self.dropout(x)  # --dropout
        x = residual + x
        if self.variant == 'aiayn' or self.variant == 'xlm' or self.variant == 'bart':
            x = self.norm3(x)

        return x, {'self_attn': final_self_attn_incr_state}

    def reorder_incremental_state(
        self, incremental_state: DecoderLayerIncrState, inds: torch.Tensor
    ) -> Dict[str, dict]:
        """
        Reorder all incremental-state tensors for this layer.
        """
        attn_types = {'self_attn': self.self_attention}
        return {
            attn_type: attn.reorder_incremental_state(
                incremental_state[attn_type], inds
            )
            for attn_type, attn in attn_types.items()
        }

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

DecoderIncrState = Dict[int, Dict[str, Dict[str, torch.Tensor]]]

class BaseTransformerDecoder(nn.Module, ABC):
    """
    Implements functionality common to all transformer decoder variants. Not intended to
    be instantiated directly.

    For a (Vaswani 2017) style encoder-decoder transformer, use ``TransformerDecoder``. For a GPT-style decoder-only transformer, use ``TransformerDecoderOnly``.

    Subclasses are required to implement ``forward``. In your ``forward`` implementation, you can call ``forward_embedding`` to get embeddings for the input tokens and ``forward_layers`` to pass those embeddings sequentially through each layer.

    Subclasses can optionally override ``__init__``, ``build_layer``, and
    ``build_layers`` to customize subcomponents. In particular, ``build_layer`` can be used to instantiate heterogeneous layers (e.g. every other layer being a different type).
    """

    def __init__(
        self,
        opt: Opt,
        embedding: nn.Embedding,
        dictionary: DictionaryAgent,
        n_positions: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]

        self.embedding_size = opt['embedding_size']
        self.ffn_size = opt['ffn_size']
        self.n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        self.n_heads = opt['n_heads']
        self.dim = self.embedding_size
        self.activation = opt.get('activation', 'relu')
        self.variant = opt.get('variant', 'aiayn')

        self.embeddings_scale = opt.get('embeddings_scale', True)
        self.dropout = nn.Dropout(p=opt.get('dropout', 0.0))  # --dropout

        self.n_positions = default(n_positions, get_n_positions_from_options(opt))
        self.out_dim = self.embedding_size
        assert (
            self.embedding_size % self.n_heads == 0
        ), 'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        if (
            self.variant == 'xlm'
            or self.variant == 'prelayernorm'
            or self.variant == 'bart'
        ):
            self.norm_embeddings = torch.nn.LayerNorm(self.dim, eps=LAYER_NORM_EPS)
            if self.variant == 'xlm':
                warn_once(
                    'DEPRECATED: XLM should only be used for backwards compatibility, '
                    'as it involves a less-stable layernorm operation.'
                )
        elif self.variant == 'aiayn':
            pass
        else:
            raise ValueError("Can't handle --variant {}".format(self.variant))

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(self.n_positions, self.embedding_size)
        if not opt.get('learn_positional_embeddings', False):
            create_position_codes(
                self.n_positions,
                self.embedding_size,
                out=self.position_embeddings.weight,
            )
        else:
            nn.init.normal_(
                self.position_embeddings.weight, 0, self.embedding_size**-0.5
            )

        # build the model
        self.layers = self.build_layers()

    def build_layers(self) -> nn.ModuleList:
        """
        Instantiates all layers. Called only once during __init__.

        Additional setup common to all layers, such as checkpoint wrapping, can be done
        here.
        """
        layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer = self.build_layer(index=i)
            if self.opt.get('checkpoint_activations'):
                layer = checkpoint_wrapper(layer)
            layers.append(fsdp_wrap(layer))  # type: ignore
        return layers

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        :param int index:
            Index of current layer.
        """
        return BaseTransformerDecoderLayer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        raise NotImplementedError

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch, seqlen] input:
            The target input IDs
        :param LongTensor[batch, seqlen] positions:
            Positions for input IDs. If None, computes defaults.
        :param LongTensor[batch, seqlen] segments:
            Segment IDs for extra embedding features. If None, not used.

        :return (tensor, mask):
            embedded input and mask
        """
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = self.norm_embeddings(tensor)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        if self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        return tensor

    def forward_layers(
        self, tensor: torch.Tensor, *extra_args, incr_state: DecoderIncrState, **kwargs
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param extra_args:
            any number of positional arguments to be passed to each layer
        :param incr_state:
            Dict mapping layer_idx to incremental state
        :param kwargs:
            any number of keyword (named) arguments to be passed to each layer

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, *extra_args, incr_state=incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    tensor, *extra_args, incr_state=incr_state.get(idx), **kwargs
                )

        return tensor, new_incr_state

    def _apply_model_parallel(
        self, tensor: torch.Tensor, *extra_args, incr_state: DecoderIncrState
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Pipeline application of model parallelism.
        """
        chunks = PipelineHelper.split((tensor, *extra_args, incr_state))
        work_items = PipelineHelper.schedule_work_items(self.layers, chunks)

        new_incr_state = {i: [] for i, _ in enumerate(self.layers)}

        for chunk_idx, layer_nos, next_device in work_items:
            s_tensor, *s_extra_args, s_incr_state = chunks[chunk_idx]
            for layer_no in layer_nos:
                s_tensor, nis = self.layers[layer_no](
                    s_tensor, *s_extra_args, incr_state=s_incr_state.get(layer_no)
                )
                new_incr_state[layer_no].append(nis)
            # don't move incr state, it's always on the correct device
            s_layer_args = PipelineHelper.chunk_to(
                (s_tensor, *s_extra_args), next_device
            )
            chunks[chunk_idx] = (*s_layer_args, s_incr_state)

        tensor_out = PipelineHelper.join([c[0] for c in chunks])
        new_incr_state = {
            layer_no: PipelineHelper.join(pieces)
            for layer_no, pieces in new_incr_state.items()
        }

        return tensor_out, new_incr_state

class TransformerDecoder(BaseTransformerDecoder):
    """
    Transformer Decoder module.

    For documentation on parameters that are take directly from opt,
    see parlai/agents/transformer/transformer.py

    :param opt: ParlAI-parsed options.
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def build_layer(self, index: int) -> BaseTransformerDecoderLayer:
        """
        Instantiate a single layer. Called n_layers times during __init__.

        Overridden to allow swapping out of the layer class at instantiation.

        :param int index:
            Index of current layer.
        """
        return self.swappables.layer(  # type: ignore
            self.opt,
            attention_dropout=self.opt.get('attention_dropout', 0.0),
            relu_dropout=self.opt.get('relu_dropout', 0.0),
            dropout=self.opt.get('dropout', 0.0),
            activation=self.activation,
            variant=self.variant,
        )

    def forward(
        self,
        input: torch.Tensor,
        encoder_state: Tuple[torch.Tensor, torch.Tensor],
        incr_state: Optional[DecoderIncrState] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecoderIncrState]:
        """
        Forward pass.

        :param LongTensor[batch,seqlen] input:
            The decoder inputs (partial or full decoded token IDs).
        :param encoder_state:
            Output from the encoder module forward pass.
        :param incr_state:
            The incremental state: a dictionary whose keys index the layers and whose
            values contain the incremental state for each layer.
        """
        encoder_output, encoder_mask = encoder_state

        seq_len = input.size(1)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=input.device
        ).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            input = input[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        tensor = self.forward_embedding(input, positions, **kwargs)

        tensor = self.dropout(tensor)  # --dropout

        tensor, new_incr_state = self.forward_layers(
            tensor, encoder_output, encoder_mask, incr_state=incr_state, **kwargs
        )

        if self.variant == 'prelayernorm':
            tensor = self.norm_embeddings(tensor)

        return tensor, new_incr_state

class TransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TransformerConfig
    # todo

class TransformerModel(TransformerPreTrainedModel):
    """
    Implements a full generator model, with one encoder and one decoder.
    """

    @classmethod
    def build_encoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        reduction_type='mean',
        encoder_class: Type[TransformerEncoder] = TransformerEncoder,
        **kwargs,
    ) -> TransformerEncoder:
        return encoder_class(
            opt=opt,
            embedding=embedding,
            vocabulary_size=len(dictionary),
            padding_idx=padding_idx,
            reduction_type=reduction_type,
            **kwargs,
        )

    @classmethod
    def build_decoder(
        cls,
        opt,
        embedding=None,
        decoder_class: Type[TransformerDecoder] = TransformerDecoder,
        **kwargs,
    ) -> TransformerDecoder:
        return decoder_class(opt=opt, embedding=embedding, **kwargs)

    def __init__(self, opt: Opt, dictionary: DictionaryAgent, **kwargs):
        self.pad_idx = dictionary[dictionary.null_token]
        self.start_idx = dictionary[dictionary.start_token]
        self.end_idx = dictionary[dictionary.end_token]
        super().__init__(self.pad_idx, self.start_idx, self.end_idx, **kwargs)
        self.opt = opt
        self.embeddings = create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.encoder = self.build_encoder(
            opt,
            dictionary,
            self.embeddings,
            self.pad_idx,
            reduction_type=None,
            encoder_class=self.swappables.encoder,  # type: ignore
        )
        self.decoder = self.build_decoder(
            opt,
            embedding=self.embeddings,
            decoder_class=self.swappables.decoder,  # type: ignore
            dictionary=dictionary,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states.

        See ``TorchGeneratorModel.reorder_encoder_states`` for a description.
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        if mask is not None:
            mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(
        self, incremental_state: Dict[int, dict], inds: torch.Tensor
    ) -> Dict[int, dict]:
        """
        Reorder the decoder incremental state.

        See ``TorchGeneratorModel.reorder_decoder_incremental_state`` for a description.

        Here, incremental_state is a dict whose keys are layer indices and whose values
        are dicts containing the incremental state for that layer.
        """
        return {
            idx: layer.reorder_incremental_state(incremental_state[idx], inds)
            for idx, layer in enumerate(self.decoder.layers)
        }

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        # compatibility with fairseq: fairseq sometimes reuses BOS tokens and
        # we need to force their probability of generation to be 0.
        output[:, :, self.start_idx] = neginf(output.dtype)
        return output
