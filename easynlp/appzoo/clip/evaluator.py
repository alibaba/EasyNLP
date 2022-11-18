# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
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

import time
import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from ...utils import losses
from ...utils.logger import logger
from ...core.evaluator import Evaluator

class CLIPEvaluator(Evaluator):
    
    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)
        self.metrics = ["accuracy", "f1"]
        self.before=0.0

    def evaluate(self, model):
        model.eval()
        total_spent_time = 0.0
        image_embeds_all=[]
        text_embeds_all=[]
        for _step, batch in enumerate(self.valid_loader):
            infer_start_time = time.time()
            with torch.no_grad():
                outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time
            image_embeds_all.append(outputs['image_embeds'])
            text_embeds_all.append(outputs['text_embeds'])
        image_embeds_tensor=torch.cat(image_embeds_all,dim=0)
        text_embeds_tensor=torch.cat(text_embeds_all,dim=0)
        query_len=text_embeds_tensor.size()[0]
        agreement=text_embeds_tensor@image_embeds_tensor.t()
        agreement_size=agreement.size()
        r1_stat, r5_stat, r10_stat = 0, 0, 0
        for idx in range(0,agreement_size[0]):
            tmp=agreement[idx].detach()
            reordered,ridx=torch.sort(tmp,descending=True)
            if idx in ridx[:1]:
                r1_stat+=1
            if idx in ridx[:5]:
                r5_stat+=1
            if idx in ridx[:10]:
                r10_stat+=1
        r1, r5, r10 = r1_stat * 1.0 / query_len, r5_stat * 1.0 / query_len, r10_stat * 1.0 / query_len
        mean_recall = (r1 + r5 + r10) / 3.0
        result = [mean_recall, r1, r5, r10]
        result = [item * 100 for item in result]
        print('r1_num:'+str(r1_stat),'r5_num:'+str(r5_stat),'r10_num:'+str(r10_stat),'query_num:'+str(query_len))
        print('r1(%):'+str(result[1]),'r5(%):'+str(result[2]),'r10(%):'+str(result[3]),'mean_recall(%):'+str(result[0]))
        logger.info("Inference time = {:.2f}s, [{:.4f} ms / sample] ".format(
            total_spent_time, total_spent_time * 1000 / query_len))
        eval_outputs = list()
        eval_outputs.append(("mean_recall", mean_recall))
        return eval_outputs