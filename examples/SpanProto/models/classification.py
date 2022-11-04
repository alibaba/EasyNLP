# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 10:54 am.
# @Author  : JianingWang
# @File    : classification.py
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import RobertaModel
from transformers.activations import ACT2FN
from transformers.models.electra import ElectraModel
from transformers.models.roformer import RoFormerModel
from transformers.models.albert import AlbertModel
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.deberta_v2 import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta import RobertaPreTrainedModel
from transformers.models.bert.modeling_bert import BertForSequenceClassification

PRETRAINED_MODEL_MAP = {
    'bert': BertPreTrainedModel,
    'deberta-v2': DebertaV2PreTrainedModel,
    'roberta': RobertaPreTrainedModel
}


class BertPooler(nn.Module):
    def __init__(self, hidden_size, hidden_act, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # self.activation = nn.Tanh()
        self.activation = ACT2FN[hidden_act]
        # self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        return x


def build_cls_model(config):
    BaseClass = PRETRAINED_MODEL_MAP[config.model_type]

    class BertForClassification(BaseClass):

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.config = config
            self.model_type = config.model_type
            self.problem_type = config.problem_type

            if self.model_type == 'bert':
                self.bert = BertModel(config)
            elif self.model_type == 'albert':
                self.albert = AlbertModel(config)
            # elif self.model_type == 'chinesebert':
            #     self.bert = ChineseBertModel(config)
            elif self.model_type == 'roformer':
                self.roformer = RoFormerModel(config)
            elif self.model_type == 'electra':
                self.electra = ElectraModel(config)
            elif self.model_type == 'deberta-v2':
                self.deberta = DebertaV2Model(config)
            elif self.model_type == 'roberta':
                self.roberta = RobertaModel(config)
            self.pooler = BertPooler(config.hidden_size, config.hidden_act, config.hidden_dropout_prob)
            if hasattr(config, 'cls_dropout_rate'):
                cls_dropout_rate = config.cls_dropout_rate
            else:
                cls_dropout_rate = config.hidden_dropout_prob
            self.dropout = nn.Dropout(cls_dropout_rate)
            add_feature_dims = config.additional_feature_dims if hasattr(config, 'additional_feature_dims') else 0
            # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            cls_hidden = config.hidden_size + add_feature_dims
            if hasattr(config, 'is_relation_task'):
                cls_hidden = config.hidden_size * 2
            self.classifier = nn.Linear(cls_hidden, config.num_labels)

            self.init_weights()

        def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                pseudo_label=None,
                pinyin_ids=None,
                additional_features=None
        ):
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            logits, outputs = None, None
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'position_ids': position_ids,
                      'head_mask': head_mask, 'inputs_embeds': inputs_embeds, 'output_attentions': output_attentions,
                      'output_hidden_states': output_hidden_states, 'return_dict': return_dict, 'pinyin_ids': pinyin_ids}
            inputs = {k: v for k, v in inputs.items() if v is not None}
            if self.model_type == 'chinesebert':
                outputs = self.bert(**inputs)
            elif self.model_type == 'bert':
                outputs = self.bert(**inputs)
            elif self.model_type == 'albert':
                outputs = self.albert(**inputs)
            elif self.model_type == 'electra':
                outputs = self.electra(**inputs)
            elif self.model_type == 'roformer':
                outputs = self.roformer(**inputs)
            elif self.model_type == 'deberta-v2':
                outputs = self.deberta(**inputs)
            elif self.model_type == 'roberta':
                outputs = self.roberta(**inputs)

            if hasattr(self.config, 'is_relation_task'):
                w = torch.logical_and(input_ids >= min(self.config.start_token_ids), input_ids <= max(self.config.start_token_ids))
                start_index = w.nonzero()[:, 1].view(-1, 2)
                pooler_output = torch.cat([torch.cat([x[y[0], :], x[y[1], :]]).unsqueeze(0) for x, y in zip(outputs.last_hidden_state, start_index)])
                # pooler_output = torch.cat([torch.cat([z, x[y[0], :], x[y[1], :]]).unsqueeze(0) for x, y, z in zip(outputs.last_hidden_state, start_index, outputs.last_hidden_state[:, 0])])

            elif 'pooler_output' in outputs:
                pooler_output = outputs.pooler_output
            else:
                pooler_output = self.pooler(outputs[0])
            pooler_output = self.dropout(pooler_output)
            # pooler_output = self.LayerNorm(pooler_output)
            if additional_features is not None:
                pooler_output = torch.cat((pooler_output, additional_features), dim=1)
            logits = self.classifier(pooler_output)

            loss = None
            if labels is not None:
                if self.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))
                # elif self.problem_type in ["single_label_classification"] or hasattr(self.config, 'is_relation_task'):
                else:
                    # loss_fct = FocalLoss()
                    loss_fct = CrossEntropyLoss()
                    if pseudo_label is not None:
                        train_logits, pseudo_logits = logits[pseudo_label > 0.9], logits[pseudo_label < 0.1]
                        train_labels, pseudo_labels = labels[pseudo_label > 0.9], labels[pseudo_label < 0.1]
                        train_loss = loss_fct(train_logits.view(-1, self.num_labels), train_labels.view(-1)) if train_labels.nelement() else 0
                        pseudo_loss = loss_fct(pseudo_logits.view(-1, self.num_labels), pseudo_labels.view(-1)) if pseudo_labels.nelement() else 0
                        loss = 0.9 * train_loss + 0.1 * pseudo_loss
                    else:
                        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    return BertForClassification
