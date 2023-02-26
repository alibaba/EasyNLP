# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and The HuggingFace Inc. team, and https://github.com/autoliuweijie/FastBERT.
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
from ast import literal_eval
import torch
import torch.nn as nn
import math
import time
import numpy as np
import pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from ...modelzoo.models.geep import GEEPModel
from ...modelzoo import AutoConfig, AutoModel
from ...utils import losses
from ..application import Application
from ...core.evaluator import Evaluator
from ...utils.logger import logger

# The GPU hash table used in our GEEP paper will release soon.

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num
        self.linear_layers = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size) for _ in range(3)
            ])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)
    def forward(self, key, value, query):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size
        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)
        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, hidden_size)
        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size)) 
        # scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output

class GEEPClassifier(nn.Module):
    """
    Classifiers for early exit.
    """
    def __init__(self, input_size, labels_num):
        super(GEEPClassifier, self).__init__()
        self.input_size = input_size
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.labels_num = labels_num
        self.pooling = "first"
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, 0.5)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)
    
    def forward(self, hidden):
        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_atten(hidden, hidden, hidden)
        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=-1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]
        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        return logits

def attr_set(classifiers,key,val):
    """
    Load weight for classifiers.
    """
    attr=key.split('.')
    rest_attr='.'.join(attr[2:])
    target_classifier=classifiers[int(attr[1])]
    if rest_attr=='output_layer_0.weight':
        target_classifier.output_layer_0.weight.data=val 
        return
    if rest_attr=='output_layer_0.bias':
        target_classifier.output_layer_0.bias.data=val 
        return
    if rest_attr=='self_atten.linear_layers.0.weight':
        target_classifier.self_atten.linear_layers[0].weight.data=val
        return
    if rest_attr=='self_atten.linear_layers.0.bias':
        target_classifier.self_atten.linear_layers[0].bias.data=val
        return
    if rest_attr=='self_atten.linear_layers.1.weight':
        target_classifier.self_atten.linear_layers[1].weight.data=val
        return
    if rest_attr=='self_atten.linear_layers.1.bias':
        target_classifier.self_atten.linear_layers[1].bias.data=val
        return
    if rest_attr=='self_atten.linear_layers.2.weight':
        target_classifier.self_atten.linear_layers[2].weight.data=val
        return
    if rest_attr=='self_atten.linear_layers.2.bias':
        target_classifier.self_atten.linear_layers[2].bias.data=val
        return
    if rest_attr=='self_atten.final_linear.weight':
        target_classifier.self_atten.final_linear.weight.data=val
        return
    if rest_attr=='self_atten.final_linear.bias':
        target_classifier.self_atten.final_linear.bias.data=val
        return
    if rest_attr=='output_layer_1.weight':
        target_classifier.output_layer_1.weight.data=val
        return
    if rest_attr=='output_layer_1.bias':
        target_classifier.output_layer_1.bias.data=val
        return
    if rest_attr=='output_layer_2.weight':
        target_classifier.output_layer_2.weight.data=val
        return
    if rest_attr=='output_layer_2.bias':
        target_classifier.output_layer_2.bias.data=val
        return
    raise Exception(key)

class GEEPClassification(Application):
    """
    GEEPClassification: a hybrid architecture including a BERT-ish backbone and multiple early-exit classifiers.
    You should provide following user_defined_parameters:
    user_defined_parameters['geep_exit_num']: Required for training. The number of early-exit classifiers, these classifiers receive each Transformer Layer output from bootom to top.
    user_defined_parameters['geep_threshold']: Required for inference, between 0 and 1. It is the threshold for the normalized cross entropy. Smaller value for higher accuracy and slower inference time.
    """
    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, user_defined_parameters={},**kwargs):
        # Map to cpu device when gpu is not available
        instance=GEEPClassification(None,user_defined_parameters)
        instance.mode='inference'
        try:
            instance.threshold=float(user_defined_parameters['geep_threshold'])
        except:
            instance.threshold=0
        instance.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        if 'num_labels' in kwargs:
            instance.config.num_labels = kwargs['num_labels']
        instance.exit_num=instance.config.exit_num
        try:
            model_state_dict = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin')
        except RuntimeError:
            model_state_dict = torch.load(pretrained_model_name_or_path+'/pytorch_model.bin',
                                            map_location=torch.device('cpu'))
        state_dict_without_prefix = {}
        for key, value in model_state_dict.items():
            if 'backbone' in key:
                key=key.replace('backbone.','')
                state_dict_without_prefix[key] = value
        instance.backbone= GEEPModel.from_pretrained(pretrained_model_name_or_path,state_dict=state_dict_without_prefix)
        instance.classifiers = nn.ModuleList([
            GEEPClassifier(instance.config.hidden_size, instance.config.num_labels) for i in range(0,instance.exit_num+1)
        ])
        for key, value in model_state_dict.items():
            if 'classifiers' in key:
                attr_set(instance.classifiers,key,value)
        instance.teacher_classifier = instance.classifiers[-1]
        instance.dropout = nn.Dropout(instance.config.hidden_dropout_prob)
        return instance

    def __init__(self, pretrained_model_name_or_path,user_defined_parameters, **kwargs):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.soft_criterion = nn.KLDivLoss(reduction='batchmean')
        if pretrained_model_name_or_path is not None:
            self.mode='train'
            self.threshold=0
            self.exit_num=int(user_defined_parameters['geep_exit_num'])
            if self.exit_num<4:
                self.exit_num=4
            if self.exit_num>12:
                self.exit_num=12
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.config.exit_num=self.exit_num
            self.backbone = GEEPModel.from_pretrained(pretrained_model_name_or_path)
            if 'num_labels' in kwargs:
                self.config.num_labels = kwargs['num_labels']
            self.classifiers = nn.ModuleList([
                    GEEPClassifier(self.config.hidden_size, self.config.num_labels) for i in range(0,self.exit_num+1)
                 ])
            self.teacher_classifier = self.classifiers[-1]
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, inputs):
        if self.mode=='train':
            inputs['classifiers']=self.classifiers
            inputs['mode']=self.mode
            inputs['output_hidden_states']=True
            inputs['exit_num']=self.exit_num
            outputs = self.backbone(**inputs)
            logits_list=[]
            for hidden_states in outputs.hidden_states:
                logits = self.teacher_classifier(hidden_states)
                logits_list.append(logits)
            return {
                'hidden': hidden_states,
                'logits': logits_list[-1],
                'logits_list':logits_list,
                'predictions': torch.argmax(logits_list[-1], dim=-1),
                'probabilities': torch.softmax(logits_list[-1], dim=-1),
                'sub_hidden_states':outputs['cross_attentions']
            }
        else:
            with torch.no_grad():
                inputs['classifiers']=self.classifiers
                inputs['mode']=self.mode
                inputs['output_hidden_states']=False#no need
                inputs['exit_num']=self.exit_num
                inputs['num_labels']=self.config.num_labels
                inputs['threshold']=self.threshold
                outputs = self.backbone(**inputs)
                infer_logits=outputs['cross_attentions']
                return {
                    'logits': infer_logits,
                    'predictions': torch.argmax(infer_logits, dim=-1),
                    'probabilities': torch.softmax(infer_logits, dim=-1)
                }

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        loss1=0
        loss_list=[]
        teacher_logit=None
        max_len=len(forward_outputs['logits_list'])
        for i in range(0,max_len):
            one_logit=forward_outputs['logits_list'][i]
            teacher_logit=one_logit
            tmp_loss=self.criterion(self.softmax(one_logit.view(-1, self.config.num_labels)), label_ids.view(-1))
            loss_list.append(tmp_loss)
            loss1+=tmp_loss
        teacher_probs = nn.functional.softmax(teacher_logit, dim=1).detach()#stop gradient
        loss2 = 0
        for i in range(0,max_len-1):
            one_hidden_state=forward_outputs['sub_hidden_states'][i]
            student_logits = self.classifiers[i](one_hidden_state).view(-1, self.config.num_labels)
            loss2 += self.soft_criterion(self.softmax(student_logits), teacher_probs) 
        return {'loss': loss1+loss2}