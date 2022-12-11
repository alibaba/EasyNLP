import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from ..application import Application
from ...modelzoo import AutoConfig, AutoModel


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """

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

class InformationExtractionModel(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()

        if kwargs.get('from_config'):
            # for evaluation and prediction
            self.config = kwargs.get('from_config')
            self.backbone = AutoModel.from_config(self.config)
        else:
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)

        self.hidden_size = self.config.hidden_size

        self.ent_type_size = 1
        self.inner_dim = 64
        self.RoPE = True

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(self.hidden_size, self.ent_type_size * 2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def forward(self, inputs):

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        context_outputs = self.backbone(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state
        outputs = self.dense_1(last_hidden_state)
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
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # logits[:, None] 增加一个维度
        #logits.shape=[2,1,512,512]

        mask = torch.triu(attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))
        
        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            topk = torch.topk(prob.view(batch_size, self.ent_type_size, -1), 50, dim=-1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'logits': logits,
            'topk_probs': topk.values,
            'topk_indices': topk.indices
        }

    def compute_loss(self, forward_outputs, label_ids, **kwargs):

        input_ids = forward_outputs['input_ids']
        attention_mask = forward_outputs['attention_mask']
        logits = forward_outputs['logits']

        mask = torch.triu(attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))
        y_pred = logits - (1-mask.unsqueeze(1))*1e12
        y_true = label_ids.view(input_ids.shape[0] * self.ent_type_size, -1)
        y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_pred, y_true)

        return {'loss': loss}
