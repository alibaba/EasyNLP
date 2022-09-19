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

import os
import re
import json
import math
import time
import timeit
import string
import collections
import torch
from torch import nn
import torch.utils.data.dataloader as DataLoader
from tqdm import tqdm, trange
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, roc_auc_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

from ...utils import losses
from ...utils.logger import logger
from ...core.evaluator import Evaluator
from ...fewshot_learning.fewshot_evaluator import PromptEvaluator, CPTEvaluator


class MachineReadingComprehensionEvaluator(object):

    def __init__(self, valid_dataset, user_defined_parameters, **kwargs):
        eval_batch_size = kwargs.get('eval_batch_size', 32)
        self.valid_dataset = valid_dataset
        self.valid_loader = DataLoader.DataLoader(self.valid_dataset,
                                                  batch_size=eval_batch_size,
                                                  shuffle=False,
                                                  collate_fn=valid_dataset.batch_fn)
        self.best_valid_score = float('-inf')

    def evaluate(self, model):
        model.eval()
        acc_sum, n = 0.0, 0

        total_spent_time = 0.0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for _step, batch in enumerate(tqdm(self.valid_loader)):
            # print("evaluation batch_unique_id_shape: ", np.array(batch["unique_id"]).shape)
            # print("evaluation batch_tokens_shape: ", np.array(batch["tokens"]).shape)
            # print("evaluation batch_tokens: ", np.array(batch["tokens"]))
            # print("evaluation batch_tok_to_orig_index_shape: ", np.array(batch["tok_to_orig_index"]).shape)
            # print("evaluation batch_tok_to_orig_index: ", np.array(batch["tok_to_orig_index"]))
            # print("evaluation batch_token_is_max_context_shape: ", np.array(batch["token_is_max_context"]).shape)
            # print("evaluation batch_token_is_max_context: ", np.array(batch["token_is_max_context"]))
            # print("evaluation batch_input_ids_shape: ", np.array(batch["input_ids"]).shape)
            # print("evaluation batch_attention_mask_shape: ", np.array(batch["attention_mask"]).shape)
            # print("evaluation batch_token_type_ids_shape: ", np.array(batch["token_type_ids"]).shape)
            # print("evaluation batch_start_position_shape: ", np.array(batch["start_position"]).shape)
            # print("evaluation batch_end_position_shape: ", np.array(batch["end_position"]).shape)
            # print("evaluation batch_label_ids_shape: ", np.array(batch["label_ids"]).shape)
            try:
                batch = {
                    # key: val.cuda() if isinstance(val, torch.Tensor) else val
                    # for key, val in batch.items()
                    key: val.to(device) if isinstance(val, torch.Tensor) else val
                    for key, val in batch.items()
                }
            except RuntimeError:
                batch = {key: val for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                start_position = batch.pop("start_position")
                end_position = batch.pop("end_position")
                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": batch["token_type_ids"]
                }
                outputs = model(inputs)
                # print("evaluation outputs_start_logits_shape: ", outputs["start_logits"].shape)
                # print("evaluation outputs_end_logits_shape: ", outputs["end_logits"].shape)
                # print("evaluation outputs_predictions_shape: ", outputs["predictions"].shape)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]
            # print("evaluation outputs_start_logits_max_shape: ", start_logits.argmax(-1).shape)
            # print("evaluation outputs_end_logits_max_shape: ", end_logits.argmax(-1).shape)
            # print("evaluation outputs_start_logits: ", str(np.array(start_logits.cpu())))
            # print("evaluation outputs_start_logits_max: ", str(np.array(start_logits.argmax(-1).cpu())))
            # print("evaluation outputs_start_position: ", str(np.array(start_position.cpu())))
            acc_sum_start = (start_logits.argmax(-1) == start_position).float().sum().item()
            acc_sum_end = (end_logits.argmax(-1) == end_position).float().sum().item()
            # print("evaluation acc_sum_start: ", acc_sum_start)
            # print("evaluation acc_sum_end: ", acc_sum_end)
            acc_sum += (acc_sum_start + acc_sum_end)
            # print("evaluation length: ", len(start_position))
            n += len(start_position)

        eval_outputs = list()
        acc = acc_sum / (2 * n)
        logger.info("Accuracy: {}".format(acc))
        eval_outputs.append(("accuracy", acc))

        # model.train()
        return eval_outputs
