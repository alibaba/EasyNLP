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
# from torch import nn
# import numpy as np
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
# from sklearn.metrics import f1_score, precision_score, recall_score

# from ...utils import losses
from ...utils.logger import logger
from ...core.evaluator import Evaluator
# from ...fewshot_learning.fewshot_evaluator import PromptEvaluator, CPTEvaluator


class FrameTextGenerationEvaluator(Evaluator):

    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)
        self.metrics = ["loss"]

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        total_steps = 0
        total_samples = 0

        total_spent_time = 0.0
        for _step, batch in enumerate(self.valid_loader):
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            
            infer_start_time = time.time()
            with torch.no_grad():
                outputs,label_ids = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            logits = outputs
            tmp_loss = model.compute_loss(logits, label_ids)['loss']
        
            total_loss += tmp_loss.mean().item()
            total_steps += 1
            total_samples += self.valid_loader.batch_size
            if (_step + 1) % 100 == 0:
                logger.info(
                    "Eval: %d/%d steps finished" %
                    (_step + 1, len(self.valid_loader.dataset) // self.valid_loader.batch_size))

        logger.info("Inference time = {:.2f}s, [{:.4f} ms / sample] ".format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info("Eval loss: {}".format(eval_loss))

        eval_outputs = list()
        eval_outputs.append(("eval_loss", -eval_loss))

        return eval_outputs


