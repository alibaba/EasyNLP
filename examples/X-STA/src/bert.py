from typing import *
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertSelfAttention, BertIntermediate, BertOutput, BertEmbeddings, BertPooler, BertAttention
)
from transformers.modeling_outputs import (
    ModelOutput,
    QuestionAnsweringModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions
)

from src.utils import BatchNorm, AttentionTeacher, get_pair_entropy, ContrastiveLoss

class BertMixSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, ratio=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if ratio is not None:
            hidden_states = self.LayerNorm(ratio * hidden_states + (1-ratio) * input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertMixSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ratio=None
    ):

        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states, ratio)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertMixLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        add_attention=True,
        add_ffn=True,
        ratio=None
    ):

        if add_attention:
            attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                ratio=ratio
            )
            if not add_ffn:
                return attention_outputs
            attention_output = attention_outputs[0]
        else:
            attention_output = hidden_states
            attention_outputs = (attention_output, None)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs



class BertMixEncoder(nn.Module):
    def __init__(self, config, mix_layer):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertMixLayer(config) for _ in range(config.num_hidden_layers)])
        self.mix_layer = mix_layer

        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))

        self.f = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        raw_attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        lang_ids=None
    ):
        if isinstance(self.mix_layer, str):
            mix_layers = [int(x) for x in self.mix_layer.split(',')]
        else:
            mix_layers = [self.mix_layer]
        attention_entropy = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        all_attention_entropy = ()

        next_decoder_cache = () if use_cache else None


        attention_mask_en = attention_mask.view(-1, 2, *attention_mask.size()[1:])[:, 0]
        attention_mask_trg = attention_mask.view(-1, 2, *attention_mask.size()[1:])[:, 1]

        raw_attention_mask = raw_attention_mask[:, None, None, :]


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if i in mix_layers:
                hidden_states_en = hidden_states.view(-1, 2, hidden_states.size(-2), hidden_states.size(-1))[:, 0]
                hidden_states_trg = hidden_states.view(-1, 2, hidden_states.size(-2), hidden_states.size(-1))[:, 1]

                # trg self attention
                self_attention_output = layer_module.attention.self(
                    hidden_states_trg,
                    attention_mask_trg,
                    layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions
                )[0]

                src_lang_id = int(lang_ids[0])
                trg_lang_id = int(lang_ids[1])
                if src_lang_id == trg_lang_id:
                    hidden_states_en_convert = hidden_states_en
                else:
                    hidden_states_en_convert = hidden_states_en.detach() + self.f(torch.cat([hidden_states_en.detach(), hidden_states_trg.detach()], dim=-1))

                cross_attention_outputs = layer_module.attention.self(
                    hidden_states_trg,
                    attention_mask_trg,
                    layer_head_mask,
                    encoder_hidden_states=hidden_states_en_convert,
                    encoder_attention_mask=attention_mask_en,
                    past_key_value=past_key_value,
                    output_attentions=True
                )

                cross_attention_output = cross_attention_outputs[0]
                cross_attention_score = cross_attention_outputs[1]

                attention_entropy = get_pair_entropy(cross_attention_score)

                ratio = self.w * 0.3 + self.b

                attention_output = layer_module.attention.output(
                    ratio * cross_attention_output + (1 - ratio) * self_attention_output,
                    hidden_states_trg
                )
                
                # trg ffn
                ffn_layer_outputs_trg = layer_module(
                    attention_output,
                    attention_mask_trg,
                    layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    add_attention=False
                )
                hidden_states_trg = ffn_layer_outputs_trg[0]

                # src
                hidden_states_en = layer_module(
                    hidden_states_en,
                    attention_mask_en,
                    layer_head_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions
                )[0]

                hidden_states = torch.stack([hidden_states_en, hidden_states_trg], dim=1)
                hidden_states = hidden_states.view(-1, *hidden_states.size()[2:])
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions
                )
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
            
            if attention_entropy is not None:
                all_attention_entropy = all_attention_entropy + (attention_entropy,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                    all_attention_entropy
                ]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertMixModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, mix_layer=7, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertMixEncoder(config, mix_layer)
        # self.encoder = BertEncoder(config)
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
        lang_ids=None
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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
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
            raw_attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lang_ids=lang_ids
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

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config, args, num_lang=2):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertMixModel(config, mix_layer=args.mix_layers if args.mix_layers is not None else args.mix_layer, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()


        self.teaching_weight = args.teaching_weight
        self.align_weight = args.align_weight
        self.consist_weight = args.consist_weight
        self.alpha = args.alpha
        self.norm = args.norm
        self.cl = args.cl
        
        if self.cl:
            self.cl_loss = ContrastiveLoss(config, temp=args.temp)
        else:
            self.mse_loss = nn.MSELoss()

        if self.teaching_weight > 0:
            self.attention_teacher = AttentionTeacher(config)
        
        if self.norm:
            self.bn = nn.ModuleList([BatchNorm(config.hidden_size) for _ in range(num_lang)])

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        query_len=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        lang_ids=None,
        return_sequence_output=False
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size(0)
        # Number of lang in one instance
        # num_lang = input_ids.size(1)

        # Flatten input
        input_ids = input_ids.view((-1, input_ids.size(-1))) # (bsz * 2, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bsz * 2, len)
        if lang_ids is not None:
            lang_ids = lang_ids.view(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bsz * 2, len)
        if query_len is not None:
            query_len = query_len.view(-1)

        outputs = self.bert(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
            lang_ids=lang_ids
        )

        sequence_output = outputs[0]

        if return_sequence_output:
            return sequence_output
        attention_entropy = outputs[-1]

        if self.norm:
            sequence_output = sequence_output.view(batch_size, 2, sequence_output.size(-2), sequence_output.size(-1))

            attention_mask_src = attention_mask.view(batch_size, 2, -1)[:, 0]
            attention_mask_trg = attention_mask.view(batch_size, 2, -1)[:, 1]

            src_lang_id = int(lang_ids[0])
            trg_lang_id = int(lang_ids[1])
            sequence_output_src = self.bn[src_lang_id](sequence_output[:, 0], attention_mask_src)
            sequence_output_trg = self.bn[trg_lang_id](sequence_output[:, 1], attention_mask_trg)
            sequence_output = torch.stack([sequence_output_src, sequence_output_trg], dim=1)
            sequence_output = sequence_output.view(batch_size * 2, *sequence_output.size()[2:])

        seq_rep = (sequence_output * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_output = sequence_output.view(batch_size, 2, sequence_output.size(-2), sequence_output.size(-1))
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if self.teaching_weight > 0:
            logits_teacher = self.attention_teacher(
                query=sequence_output[:, 1],
                key=sequence_output[:, 0],
                value=logits[:, 0].detach(),
                attention_mask=extended_attention_mask.view(batch_size, 2, *extended_attention_mask.size()[1:])[:, 0]
            )

            start_logits_t, end_logits_t = logits_teacher.split(1, dim=-1)
            start_logits_t = start_logits_t.squeeze(-1).contiguous()
            end_logits_t = end_logits_t.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.view(-1)
            end_positions = end_positions.view(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(-1)
            start_positions = start_positions.view(-1).clamp(0, ignored_index)
            end_positions = end_positions.view(-1).clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss_src = loss_fct(start_logits.view(batch_size, 2, -1)[:, 0], start_positions.view(-1, 2)[:, 0])
            end_loss_src = loss_fct(end_logits.view(batch_size, 2, -1)[:, 0], end_positions.view(-1, 2)[:, 0])

            start_loss_trg = loss_fct(start_logits.view(batch_size, 2, -1)[:, 1], start_positions.view(-1, 2)[:, 1])
            end_loss_trg = loss_fct(end_logits.view(batch_size, 2, -1)[:, 1], end_positions.view(-1, 2)[:, 1])


            loss = self.alpha * (start_loss_src + end_loss_src) / 2 \
                        + (1 - self.alpha) * (start_loss_trg + end_loss_trg) / 2 

            if self.teaching_weight > 0:
                start_loss_t = loss_fct(start_logits_t, start_positions.view(-1, 2)[:, 1])
                end_loss_t = loss_fct(end_logits_t, end_positions.view(-1, 2)[:, 1])
                loss += self.teaching_weight * (start_loss_t + end_loss_t) / 2

            loss += self.align_weight * attention_entropy[0].mean()
            
            seq_rep = seq_rep.view(batch_size, 2, -1)
            if self.cl:
                loss += self.consist_weight * self.cl_loss(seq_rep[:, 0], seq_rep[:, 1])
            else:
                loss += self.consist_weight * self.mse_loss(seq_rep[:, 0], seq_rep[:, 1])

        else:
            # predict
            start_logits = start_logits[:, 1]
            end_logits = end_logits[:, 1]
            if self.teaching_weight > 0:
                start_logits += start_logits_t
                end_logits += end_logits_t

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            return QuestionAnsweringModelOutput(
                loss=loss,
                start_logits=start_logits,
                end_logits=end_logits,
            )
