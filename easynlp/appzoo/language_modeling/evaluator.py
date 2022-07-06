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

from ...utils import losses
from ...utils.logger import logger
from ...core.evaluator import Evaluator


class LanguageModelingEvaluator(Evaluator):

    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)

        self.metrics = ["mlm_accuracy"]

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        total_spent_time = 0.0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for _step, batch in enumerate(self.valid_loader):
            batch = {
                # key: val.cuda() if isinstance(val, torch.Tensor) else val
                # for key, val in batch.items()
                key: val.to(device) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop("label_ids")
                mask_span_indices = batch.pop("mask_span_indices")
                outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert "logits" in outputs
            logits = outputs["logits"]

            for b in range(label_ids.shape[0]):
                _logits = logits[b]
                _label_ids = label_ids[b]
                _mask_span_indices = mask_span_indices[b]
                for span_indices in _mask_span_indices:
                    pred = list()
                    label = list()
                    for span_idx in span_indices:
                        pred.append(torch.argmax(_logits[span_idx]).item())
                        label.append(_label_ids[span_idx].item())
                    hit_num += (tuple(pred) == tuple(label))
                    total_num += 1
            logits = logits.view(-1, logits.size(-1))
            label_ids = label_ids.view(-1)
            indices = (label_ids != -100)
            logits = logits[indices]
            label_ids = label_ids[indices]

            tmp_loss = losses.cross_entropy(logits, label_ids)

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
        acc = hit_num / total_num
        logger.info("Accuracy: {}".format(acc))
        eval_outputs = [("accuracy", acc)]

        return eval_outputs
