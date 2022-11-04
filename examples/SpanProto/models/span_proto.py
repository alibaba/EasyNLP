# -*- coding: utf-8 -*-
# @Time    : 2022/5/20 5:30 pm.
# @Author  : JianingWang
# @File    : span_proto.py
import os
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
from typing import Union
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel


class RawGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class SinusoidalPositionEmbedding(nn.Module):

    def __init__(
            self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(y_pred, y_true, pos_loss)
    return (neg_loss + pos_loss).mean()


def multilabel_categorical_crossentropy2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred.clone()
    y_pred_pos = y_pred.clone()
    y_pred_neg[y_true>0] -= float('inf')
    y_pred_pos[y_true<1] -= float('inf')
    # y_pred_neg = y_pred - y_true * float('inf')  # mask the pred outputs of pos classes
    # y_pred_pos = y_pred - (1 - y_true) * float('inf')  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(y_pred, y_true, pos_loss)
    return (neg_loss + pos_loss).mean()

@dataclass
class GlobalPointerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    topk_probs: torch.FloatTensor = None
    topk_indices: torch.IntTensor = None
    last_hidden_state: torch.FloatTensor = None


@dataclass
class SpanProtoOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    query_spans: list = None
    proto_logits: list = None
    topk_probs: torch.FloatTensor = None
    topk_indices: torch.IntTensor = None


class SpanDetector(BertPreTrainedModel):
    def __init__(self, config):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__(config)
        self.bert = BertModel(config)
        # self.ent_type_size = config.ent_type_size
        self.ent_type_size = 1
        self.inner_dim = 64
        self.hidden_size = config.hidden_size
        self.RoPE = True

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2)  # (inner_dim * 2, ent_type_size * 2)


    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, short_labels=None):
        # with torch.no_grad():
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state # [bz, seq_len, hidden_dim]
        del context_outputs
        outputs = self.dense_1(last_hidden_state) # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[..., 1::2]
        batch_size = input_ids.shape[0]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1) # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim ** 0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None]
        # logit_mask = self.add_mask_tril(logits, mask=attention_mask)
        loss = None

        mask = torch.triu(attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))
        # mask = torch.where(mask > 0, 0.0, 1)
        if labels is not None:
            # y_pred = torch.zeros(input_ids.shape[0], self.ent_type_size, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
            # for i in range(input_ids.shape[0]):
            #     for j in range(self.ent_type_size):
            #         y_pred[i, j, labels[i, j, 0], labels[i, j, 1]] = 1
            # y_true = labels.reshape(input_ids.shape[0] * self.ent_type_size, -1)
            # y_pred = logit_mask.reshape(input_ids.shape[0] * self.ent_type_size, -1)
            # loss = multilabel_categorical_crossentropy(y_pred, y_true)
            #

            # weight = ((labels == 0).sum() / labels.sum())/5
            # loss_fct = nn.BCEWithLogitsLoss(weight=weight)
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # unmask_labels = labels.view(-1)[mask.view(-1) > 0]
            # loss = loss_fct(logits.view(-1)[mask.view(-1) > 0], unmask_labels.float())
            # if unmask_labels.sum() > 0:
            #     loss = (loss[unmask_labels > 0].mean()+loss[unmask_labels < 1].mean())/2
            # else:
            #     loss = loss[unmask_labels < 1].mean()
            # y_pred = logits.view(-1)[mask.view(-1) > 0]
            # y_true = labels.view(-1)[mask.view(-1) > 0]
            # loss = multilabel_categorical_crossentropy2(y_pred, y_true)
            # y_pred = logits - torch.where(mask > 0, 0.0, float('inf')).unsqueeze(1)
            y_pred = logits - (1-mask.unsqueeze(1))*1e12
            y_true = labels.view(input_ids.shape[0] * self.ent_type_size, -1)
            y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)

        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            topk = torch.topk(prob.view(batch_size, self.ent_type_size, -1), 50, dim=-1)


        return GlobalPointerOutput(
            loss=loss,
            topk_probs=topk.values,
            topk_indices=topk.indices,
            last_hidden_state=last_hidden_state
        )


class SpanProto(nn.Module):
    def __init__(self, config):
        '''
        word_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.config = config
        self.output_dir = "./outputs"
        # self.predict_dir = self.predict_result_path(self.output_dir)
        self.drop = nn.Dropout()
        self.global_span_detector = SpanDetector(config=self.config) # global span detector
        self.projector = nn.Sequential( # projector
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Sigmoid(),
            # nn.LayerNorm(2)
        )
        self.tag_embeddings = nn.Embedding(2, self.config.hidden_size) # tag for labeled / unlabeled span set
        # self.tag_mlp = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.max_length = 64
        self.margin_distance = 6.0
        self.global_step = 0

    def predict_result_path(self, path=None):
        if path is None:
            predict_dir = os.path.join(
                self.output_dir, "{}-{}-{}".format(self.mode, self.num_class, self.num_example), "predict"
            )
        else:
            predict_dir = os.path.join(
                path, "predict"
            )

        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
        return predict_dir


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        config = kwargs.pop("config", None)
        model = SpanProto(config=config)
        # load bert
        model.global_span_detector = SpanDetector.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs
        )
        return model

    # @classmethod
    # def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
    #     self.global_span_detector.resize_token_embeddings(new_num_tokens)

    def __dist__(self, x, y, dim, use_dot=False):
        # x: [1, class_num, hidden_dim], y: [span_num, 1, hidden_dim]
        # x - y: [span_num, class_num, hidden_dim]
        # (x - y)^2.sum(2): [span_num, class_num]
        if use_dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __get_proto__(self, support_emb: torch, support_span: list, support_span_type: list, use_tag=False):
        '''
        support_emb: [n', seq_len, dim]
        support_span: [n', m, 2] e.g. [[[3, 6], [12, 13]], [[1, 3]], ...]
        support_span_type: [n', m] e.g. [[2, 1], [5], ...]
        '''
        prototype = list() # proto type
        all_span_embs = list() # save span embedding
        all_span_tags = list()
        for tag in range(self.num_class):
            # tag_id = torch.Tensor([1 if tag == self.num_class else 0]).long().cuda()
            # tag_embeddings = self.tag_embeddings(tag_id).view(-1)
            tag_prototype = list() # [k, dim]
            # for each sentence in one episode
            for emb, span, type in zip(support_emb, support_span, support_span_type):
                # emb: [seq_len, dim], span: [m, 2], type: [m]
                span = torch.Tensor(span).long().cuda() # e.g. [[3, 4], [9, 11]]
                type = torch.Tensor(type).long().cuda() # e.g. [1, 4]
                try:
                    tag_span = span[type == tag] # e.g. span==[[3, 4]], tag==1
                    # span embedding
                    for (s, e) in tag_span:
                        # tag_emb = torch.cat([emb[s], emb[e - 1]]) # [2*dim]
                        tag_emb = emb[s] + emb[e] # [dim]
                        tag_prototype.append(tag_emb)
                        all_span_embs.append(tag_emb)
                        all_span_tags.append(tag)
                except:
                    tag_prototype.append(torch.randn(support_emb.shape[-1]).cuda())
                    # assert 1 > 2
            try:
                prototype.append(torch.mean(torch.stack(tag_prototype), dim=0))
            except:
                # print("the class {} has no span".format(tag))
                prototype.append(torch.randn(support_emb.shape[-1]).cuda())
                # assert 1 > 2
        all_span_embs = torch.stack(all_span_embs).detach().cpu().numpy().tolist()

        return torch.stack(prototype), all_span_embs, all_span_tags # [num_class + 1, dim]


    def __batch_dist__(self, prototype: torch, query_emb: torch, query_spans: list, query_span_type: Union[list, None]):
        '''
        use for classifying for query set
        '''
        all_logits = list()
        all_types = list()
        visual_all_types, visual_all_embs = list(), list()
        # num = 0
        for emb, span in zip(query_emb, query_spans):
            # assert len(span) == len(query_span_type[num]), "span={}\ntype{}".format(span, query_span_type[num])
            # print("len(span)={}, len(type)= {}".format(len(span), len(query_span_type[num])))
            span_emb = list() # [m', dim]
            try:
                for (s, e) in span:
                    tag_emb = emb[s] + emb[e]  # [dim]
                    span_emb.append(tag_emb)
            except:
                span_emb = []
            if len(span_emb) != 0:
                span_emb = torch.stack(span_emb) # [span_num, dim]
                # distance between span and prototype
                logits = self.__dist__(prototype.unsqueeze(0), span_emb.unsqueeze(1), 2) # [span_num, num_class]
                # pred_types = torch.argmax(logits, -1).detach().cpu().numpy().tolist()
                with torch.no_grad():
                    pred_dist, pred_types = torch.max(logits, -1)
                    pred_dist = torch.pow(-1 * pred_dist, 0.5)
                    # print("pred_dist=", pred_dist)
                    pred_types[pred_dist > self.margin_distance] = self.num_class
                    pred_types = pred_types.detach().cpu().numpy().tolist()

                all_logits.append(logits)
                all_types.append(pred_types)
                visual_all_types.extend(pred_types)
                visual_all_embs.extend(span_emb.detach().cpu().numpy().tolist())
            else:
                all_logits.append([])
                all_types.append([])
            # num += 1

        if query_span_type is not None:
            # query_span_type: [n', m]
            try:
                all_type = torch.Tensor([type for types in query_span_type for type in types]).long().cuda() # [span_num]
                loss = nn.CrossEntropyLoss()(torch.cat(all_logits, 0), all_type)
            except:
                all_logit, all_type = list(), list()
                for logits, types in zip(all_logits, query_span_type):
                    if len(logits) != 0 and len(types) != 0 and len(logits) == len(types):
                        # print("len(logits)=", len(logits))
                        # print("len(types)=", len(types))
                        # print("logits=", logits)
                        all_logit.append(logits)
                        all_type.extend(types)
                # print("all_logit=", all_logit)
                if len(all_logit) != 0:
                    all_logit = torch.cat(all_logit, 0)
                    all_type = torch.Tensor(all_type).long().cuda()
                    # print("len(all_logits)=", len(all_logits))
                    # print("len(query_span_type)=", len(query_span_type))

                    # print("types.shape=", torch.Tensor(all_type).shape)

                    # min_len = min(len(all_type), len(all_type))
                    # all_logit, all_type = all_logit[: min_len], all_type[: min_len]
                    # print("logits.shape=", all_logit.shape)
                    # print('all_type=', all_type)
                    loss = nn.CrossEntropyLoss()(all_logit, all_type)
                else:
                    loss = 0.


        else:
            loss = None
        all_logits = [i.detach().cpu().numpy().tolist() for i in all_logits if len(i) != 0]
        return loss, all_logits, all_types, visual_all_types, visual_all_embs


    def __batch_margin__(self, prototype: torch, query_emb: torch, query_unlabeled_spans: list,
                         query_labeled_spans: list, query_span_type: list):
        '''
        margin-based learning
        '''

        # prototype: [num_class, dim], negative: [span_num, dim]
        def distance(input1, input2, p=2, eps=1e-6):
            # Compute the distance (p-norm)
            norm = torch.pow(torch.abs((input1 - input2 + eps)), p)
            pnorm = torch.pow(torch.sum(norm, -1), 1.0 / p)
            return pnorm

        unlabeled_span_emb, labeled_span_emb, labeled_span_type = list(), list(), list()
        for emb, span in zip(query_emb, query_unlabeled_spans):
            for (s, e) in span:
                tag_emb = emb[s] + emb[e]  # [dim]
                unlabeled_span_emb.append(tag_emb)

        try:
            unlabeled_span_emb = torch.stack(unlabeled_span_emb) # [span_num, dim]
            # labeled_span_emb = torch.stack(labeled_span_emb) # [span_num, dim]
            # labeled_span_type = torch.stack(labeled_span_type) # [span_num]
        except:
            return 0.

        unlabeled_dist = distance(prototype.unsqueeze(0), unlabeled_span_emb.unsqueeze(1)) # [span_num, num_class]
        # labeled_dist = distance(prototype.unsqueeze(0), labeled_span_emb.unsqueeze(1)) # [span_num, num_class]
        # labeled_type_dist = torch.gather(labeled_dist, -1, labeled_span_type.unsqueeze(1)) # [span_num, 1]
        # print(dist)
        unlabeled_output = torch.maximum(torch.zeros_like(unlabeled_dist), self.margin_distance - unlabeled_dist)
        # labeled_output = torch.maximum(torch.zeros_like(labeled_type_dist), labeled_type_dist)
        # return torch.mean(unlabeled_output) + torch.mean(labeled_output)
        return torch.mean(unlabeled_output)


    def forward(
            self,
            episode_ids,
            support, query,
            num_class,
            num_example,
            mode=None,
            short_labels=None,
            stage:str ='train',
            path: str=None
    ):
        '''
        episode_ids: Input of the idx of each episode data. (only list)
        support: Inputs of the support set.
        query: Inputs of the query set.
        num_class: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        if stage.startswith("train"):
            self.global_step += 1
        self.num_class = num_class # N-way
        self.num_example = num_example # K-shot
        # print('num_class=', num_class)
        self.mode = mode # FewNERD mode=inter/intra
        self.max_length = support['input_ids'].shape[1]
        support_inputs, support_attention_masks, support_type_ids = \
            support['input_ids'], support['attention_mask'], support['token_type_ids'] # torch, [n, seq_len]
        query_inputs, query_attention_masks, query_type_ids = \
            query['input_ids'], query['attention_mask'], query['token_type_ids'] # torch, [n, seq_len]
        support_labels = support['labels'] # torch,
        query_labels = query['labels'] # torch,
        # global span detector: obtain all mention span and loss
        support_detector_outputs = self.global_span_detector(
            support_inputs, support_attention_masks, support_type_ids, support_labels, short_labels=short_labels
        )
        query_detector_outputs = self.global_span_detector(
            query_inputs, query_attention_masks, query_type_ids, query_labels, short_labels=short_labels
        )
        device_id = support_inputs.device.index

        # if stage == "train_span":
        if self.global_step <= 500 and stage == "train":
            # only train span detector
            return SpanProtoOutput(
                loss=support_detector_outputs.loss,
                topk_probs=query_detector_outputs.topk_probs,
                topk_indices=query_detector_outputs.topk_indices,
            )
        # obtain labeled span from the support set
        support_labeled_spans = support['labeled_spans'] # all labeled span, list, [n, m, 2], n sentence, m entity span, 2 (start / end)
        support_labeled_types = support['labeled_types'] # all labeled ent type id, list, [n, m],
        query_labeled_spans = query['labeled_spans']  # all labeled span, list, [n, m, 2], n sentence, m entity span, 2 (start / end)
        query_labeled_types = query['labeled_types']  # all labeled ent type id, list, [n, m],

        # for span, type in zip(query_labeled_spans, query_labeled_types): 
        #     assert len(span) == len(type), "span={}\ntype{}".format(span, type)

        # obtain unlabeled span from the support set
        # according to the detector, we can obtain multiple unlabeled span, which generated by the detector
        # but not labeled in n-way k-shot episode
        # support_predict_spans = self.get_topk_spans( #
        #     support_detector_outputs.topk_probs,
        #     support_detector_outputs.topk_indices,
        #     support['input_ids']
        # ) # [n, m, 2]
        # print('predicted support span num={}'.format([len(i) for i in support_predict_spans]))
        # e.g，[5, 50, 4, 43, 5, 5, 1, 50, 2, 5, 6, 4, 50, 8, 12, 28, 17]

        # we can also obtain all predicted span from the query set
        query_predict_spans = self.get_topk_spans(  #
            query_detector_outputs.topk_probs,
            query_detector_outputs.topk_indices,
            query['input_ids'],
            threshold=0.9 if stage.startswith("train") else 0.95,
            is_query=True
        )  # [n, m, 2]
        # print('predicted query span num={}'.format([len(i) for i in query_predict_spans]))


        # merge predicted span and labeled span, and generate other class for unlabeled span set
        # support_all_spans, support_span_types = self.merge_span(
        #     labeled_spans=support_labeled_spans,
        #     labeled_types=support_labeled_types,
        #     predict_spans=support_predict_spans,
        #     stage=stage
        # ) # [n, m, 2]
        # print('merged support span num={}'.format([len(i) for i in support_all_spans]))


        if stage.startswith("train"):
            query_unlabeled_spans = self.split_span(
                labeled_spans=query_labeled_spans,
                labeled_types=query_labeled_types,
                predict_spans=query_predict_spans,
                stage=stage
            )  # [n, m, 2]
            # print('merged query span num={}'.format([len(i) for i in query_all_spans]))
            query_all_spans = query_labeled_spans
            query_span_types = query_labeled_types

        else:
            query_unlabeled_spans = None
            query_all_spans, _ = self.merge_span(
                labeled_spans=query_labeled_spans,
                labeled_types=query_labeled_types,
                predict_spans=query_predict_spans,
                stage=stage
            )  # [n, m, 2]
            # query_all_spans = query_predict_spans
            query_span_types = None
            # for query_label, query_pred in zip(query_labeled_spans, query_predict_spans):
            #     print(" ==== ")
            #     print('query_labeled_spans=', query_label)
            #     print('query_predict_spans=', query_pred)

        # obtain representations of each token
        support_emb, query_emb = support_detector_outputs.last_hidden_state, \
                                 query_detector_outputs.last_hidden_state # [n, seq_len, dim]
        support_emb, query_emb = self.projector(support_emb), self.projector(query_emb) # [n, seq_len, dim]

        batch_result = dict()
        proto_losses = list()
        # batch_visual = list()
        current_support_num = 0
        current_query_num = 0
        typing_loss = None
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            id_ = episode_ids[i]

            # locate one episode and obtain the span prototype
            # [n', seq_len, dim] n' sentence in one episode
            # support_proto [num_class + 1, dim]
            support_proto, all_span_embs, all_span_tags = self.__get_proto__(
                support_emb[current_support_num: current_support_num + sent_support_num], # [n', seq_len, dim]
                support_labeled_spans[current_support_num: current_support_num + sent_support_num],  # [n', m]
                support_labeled_types[current_support_num: current_support_num + sent_support_num],  # [n', m]
            )

            # for each query, we first obtain corresponding span, and then calculate distance between it and each prototype
            # # [n', seq_len, dim] n' sentence in one episode
            proto_loss, proto_logits, all_types, visual_all_types, visual_all_embs = self.__batch_dist__(
                support_proto,
                query_emb[current_query_num: current_query_num + sent_query_num], # [n', seq_len, dim]
                query_all_spans[current_query_num: current_query_num + sent_query_num],  # [n', m]
                query_span_types[current_query_num: current_query_num + sent_query_num] if query_span_types else None,  # [n', m]
            )

            visual_data = {
                'data': all_span_embs + visual_all_embs,
                'target': all_span_tags + visual_all_types,
            }

            if stage.startswith("train"):

                margin_loss = self.__batch_margin__(
                    support_proto,
                    query_emb[current_query_num: current_query_num + sent_query_num],  # [n', seq_len, dim]
                    query_unlabeled_spans[current_query_num: current_query_num + sent_query_num],  # [n', span_num]
                    query_all_spans[current_query_num: current_query_num + sent_query_num],
                    query_span_types[current_query_num: current_query_num + sent_query_num],
                )

                proto_losses.append(proto_loss + margin_loss)

            batch_result[id_] = {
                "spans": query_all_spans[current_query_num: current_query_num + sent_query_num],
                "types": all_types,
                "visualization": visual_data
            }

            current_query_num += sent_query_num
            current_support_num += sent_support_num
        # proto_logits = torch.stack(proto_logits)
        if stage.startswith("train"):
            typing_loss = torch.mean(torch.stack(proto_losses), dim=-1)


        if not stage.startswith("train"):
            self.__save_evaluate_predicted_result__(batch_result, device_id=device_id, stage=stage, path=path)

        return SpanProtoOutput(
            loss=(support_detector_outputs.loss + typing_loss)
            if stage.startswith("train") else query_detector_outputs.loss,
        )

    def __save_evaluate_predicted_result__(self, new_result: dict, device_id: int = 0, stage="dev", path=None):
        '''
        new_result / result: {
            '(id)': { # id-th episode query
                'spans': [[[1, 4], [6, 7], xxx], ... ] # [sent_num, span_num, 2]
                'types': [[2, 0, xxx], ...] # [sent_num, span_num]
            },
            xxx
        }
        '''
        self.predict_dir = self.predict_result_path(path)
        npy_file_name = os.path.join(self.predict_dir, "{}_predictions_{}.npy".format(stage, device_id))
        result = dict()
        if os.path.exists(npy_file_name):
            result = np.load(npy_file_name, allow_pickle=True)[()]
        for episode_id, query_res in new_result.items():
            result[episode_id] = query_res
        np.save(npy_file_name, result, allow_pickle=True)


    def get_topk_spans(self, probs, indices, input_ids, threshold=0.60, low_threshold=0.1, is_query=False):
        '''
        probs: [n, m]
        indices: [n, m]
        input_texts: [n, seq_len]
        is_query: if true, each sentence must recall at least one span
        '''
        probs = probs.squeeze(1).detach().cpu() 
        indices = indices.squeeze(1).detach().cpu()
        input_ids = input_ids.detach().cpu()
        # print('probs=', probs) # [n, m]
        # print('indices=', indices) # [n, m]
        predict_span = list()
        if is_query:
            low_threshold = 0.0
        for prob, index, text in zip(probs, indices, input_ids):
            threshold_ = threshold
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            span = set()
            entity_index = index[prob >= low_threshold]
            index_ids = index_ids[prob >= low_threshold]
            while threshold_ >= low_threshold:
                for ei, entity in enumerate(entity_index):
                    p = prob[index_ids[ei]]
                    if p < threshold_:
                        break
                    start_end = np.unravel_index(entity, (self.max_length, self.max_length))
                    # print('self.max_length=', self.max_length)
                    s, e = start_end[0], start_end[1]
                    ans = text[s: e]
                    # if ans not in answer:
                    #     answer.append(ans)
                    #     topk_answer_dict[ans] = {'prob': float(prob[index_ids[ei]]), 'pos': [(s, e)]}
                    span.add((s, e))
                if len(span) <= 3:
                    threshold_ -= 0.05
                else:
                    break
            if len(span) == 0:
                span = [[0, 0]]
            span = [list(i) for i in list(span)]
            # print("prob=", prob) e.g. [0.96, 0.85, 0.04, 0.00, ...]
            # print("span=", span) e.g. [[20, 23], [11, 14]]
            predict_span.append(span)
        return predict_span


    def split_span(self, labeled_spans: list, labeled_types: list, predict_spans: list, stage: str = "train"):
        def check_similar_span(span1, span2):
            if len(span1) == 0 or len(span2) == 0:
                return False
            if span1[0] == span1[1] and span2[0] == span2[1] and abs(span1[0] - span2[0]) == 1:
                return False
            if abs(span1[0] - span2[0]) <= 1 and abs(span1[1] - span2[1]) <= 1:
                return True
            return False

        all_spans, span_types = list(), list() # [n, m]
        num = 0
        unlabeled_spans = list()
        for labeled_span, labeled_type, predict_span in zip(labeled_spans, labeled_types, predict_spans):
            unlabeled_span = list()
            # if len(all_span) != len(span_type):
            #     length = min(len(all_span), len(span_type))
            #     all_span, span_type = all_span[: length], span_type[: length]
            for span in predict_span:
                if span not in labeled_span:
                    is_remove = False
                    for span_x in labeled_span:
                        is_remove = check_similar_span(span_x, span)
                        if is_remove is True:
                            break
                    if is_remove is True:
                        continue
                    unlabeled_span.append(span)
            num += len(unlabeled_span)
            unlabeled_spans.append(unlabeled_span)
        # print("num=", num)
        return unlabeled_spans


    def merge_span(self, labeled_spans: list, labeled_types: list, predict_spans: list, stage: str = "train"):

        def check_similar_span(span1, span2):
            if len(span1) == 0 or len(span2) == 0:
                return False
            if span1[0] == span1[1] and span2[0] == span2[1] and abs(span1[0] - span2[0]) == 1:
                return False
            if abs(span1[0] - span2[0]) <= 1 and abs(span1[1] - span2[1]) <= 1:
                return True
            return False

        all_spans, span_types = list(), list() # [n, m]
        for labeled_span, labeled_type, predict_span in zip(labeled_spans, labeled_types, predict_spans):
            unlabeled_num = 0
            all_span, span_type = labeled_span, labeled_type
            if len(all_span) != len(span_type):
                length = min(len(all_span), len(span_type))
                all_span, span_type = all_span[: length], span_type[: length]
            for span in predict_span:
                if span not in all_span:
                    is_remove = False
                    for span_x in all_span:
                        is_remove = check_similar_span(span_x, span)
                        if is_remove is True:
                            break
                    if is_remove is True:
                        continue
                    all_span.append(span)
                    span_type.append(self.num_class)
                    unlabeled_num += 1
            # if self.global_step % 1000 == 0:
            #     print(" === ")
            #     print('labeled_span=', labeled_span) # [[1, 3], [12, 14], [25, 25], [7, 7]]
            #     print('predict_span=', predict_span) # [[25, 25], [1, 3], [12, 14], [7, 7]]
            if unlabeled_num == 0 and stage.startswith("train"):
                # print("unlabeled span is empty, so we randomly select one span as the unlabeled span")
                # all_span.append([0, 0])
                # span_type.append(self.num_class)
                while True:
                    random_span = np.random.randint(0, 32, 2).tolist()
                    if abs(random_span[0] - random_span[1]) > 10:
                        continue
                    random_span = [random_span[1], random_span[0]] if random_span[0] > random_span[1] else random_span
                    if random_span in all_span:
                        continue
                    all_span.append(random_span)
                    span_type.append(self.num_class)
                    break

            # if len(all_span) != len(span_type):
            #     all_span = [[0, 0]]
            #     span_type = [self.num_class]

            all_spans.append(all_span)
            span_types.append(span_type)

        return all_spans, span_types

