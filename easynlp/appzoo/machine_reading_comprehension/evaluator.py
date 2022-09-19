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
import time
import torch
import torch.utils.data.dataloader as DataLoader
from tqdm import tqdm
from ...utils.logger import logger


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
                
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]
            acc_sum_start = (start_logits.argmax(-1) == start_position).float().sum().item()
            acc_sum_end = (end_logits.argmax(-1) == end_position).float().sum().item()
            acc_sum += (acc_sum_start + acc_sum_end)
            n += len(start_position)

        eval_outputs = list()
        acc = acc_sum / (2 * n)
        logger.info("Accuracy: {}".format(acc))
        eval_outputs.append(("accuracy", acc))

        return eval_outputs
