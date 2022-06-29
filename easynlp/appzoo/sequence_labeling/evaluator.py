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

from .labeling_eval_utils import evaluate_sequence_labeling
from ...utils import losses
from ...utils.logger import logger
from ...core.evaluator import Evaluator


class SequenceLabelingEvaluator(Evaluator):

    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)

        self.metrics = ["sequence_labeling"]

    def evaluate(self, model):
        model.eval()

        def predict_sequence_labeling(raw_preds, raw_label_ids, label_enumerate_values,
                                      tok_to_orig_indexes):
            new_preds = list()
            new_labels = list()
            idx_label_map = dict({idx: value for idx, value in enumerate(label_enumerate_values)})
            for idx, (raw_pred, tok_to_orig_index) in enumerate(zip(raw_preds,
                                                                    tok_to_orig_indexes)):
                raw_label = raw_label_ids[idx]
                final_pred = list()
                final_label = list()
                prev_token_idx = -1
                for k in range(min(len(raw_pred), len(tok_to_orig_index))):
                    token_pred = raw_pred[k]
                    token_label = raw_label[k]
                    token_orig_idx = tok_to_orig_index[k]
                    if token_orig_idx == -100 or token_label == -100:
                        continue
                    if token_orig_idx == prev_token_idx:
                        continue
                    final_pred.append(idx_label_map[token_pred])
                    final_label.append(idx_label_map[token_label])
                    prev_token_idx = token_orig_idx
                raw_sequence_length = max(tok_to_orig_index) + 1
                while len(final_pred) < raw_sequence_length:
                    final_pred.append(idx_label_map[len(idx_label_map) - 1])
                    final_label.append(idx_label_map[len(idx_label_map) - 1])
                new_preds.extend(final_pred + ["O"])
                new_labels.extend(final_label + ["O"])
            return new_preds, new_labels

        total_loss = 0
        total_steps = 0
        total_samples = 0
        true_seqs = list()
        pred_seqs = list()

        total_spent_time = 0.0
        for _step, batch in enumerate(self.valid_loader):
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop("label_ids")
                tok_to_orig_index = batch.pop("tok_to_orig_index")
                outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert "logits" in outputs
            logits = outputs["logits"]

            raw_preds = torch.argmax(logits, dim=-1).tolist()
            raw_label_ids = label_ids.tolist()
            new_preds, new_labels = predict_sequence_labeling(
                raw_preds, raw_label_ids, self.valid_loader.dataset.label_enumerate_values,
                tok_to_orig_index)
            pred_seqs.extend(new_preds)
            true_seqs.extend(new_labels)
            logits = logits.view(-1, logits.size(-1))
            label_ids = label_ids.view(-1)

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

        (prec, rec, f1) = evaluate_sequence_labeling(true_seqs, pred_seqs)
        logger.info("Labeling F1:        {}".format(f1))
        logger.info("Labeling Precision: {}".format(prec))
        logger.info("Labeling Recall:     {}".format(rec))
        eval_outputs = list()
        eval_outputs.append(("labeling_f1", f1))
        eval_outputs.append(("labeling_precision", prec))
        eval_outputs.append(("labeling_recall", rec))
        return eval_outputs
