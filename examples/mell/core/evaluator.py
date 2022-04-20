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
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, \
    roc_auc_score, classification_report
from torch.utils.data import DataLoader

import losses
from utils.labeling_eval_utils import evaluate_sequence_labeling
from utils.logger import logger


class Evaluator(object):
    def __init__(self, metrics=("accuracy",)):
        self.metrics = metrics

    def evaluate(self, model, valid_loader=None,
                 valid_dataset=None, eval_batch_size=32, teacher_model=None):
        assert valid_dataset is not None or valid_loader is not None

        if valid_loader is None:
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=eval_batch_size,
                                      shuffle=False,
                                      collate_fn=valid_dataset.batch_fn)

        logger.info("=" * 10 + " Evaluate Start " + "=" * 10)
        logger.info("Eval batch size: {}".format(eval_batch_size))
        logger.info("Evaluation steps: {}".format(len(valid_loader)))

        model = model.cuda()
        model.eval()
        if teacher_model is not None:
            teacher_model = teacher_model.cuda()
            teacher_model.eval()
        if hasattr(model, "module"):
            model = model.module

        student_num_params = sum([p.nelement() for n, p in model.named_parameters()])
        logger.info("Total parameters = %s" % format(student_num_params, ","))

        evaluate_task = "none"
        if hasattr(model, "model_name"):
            if model.model_name.startswith("text_classify"):
                evaluate_task = "text_classify"
            elif model.model_name.startswith("language_modeling"):
                evaluate_task = "language_modeling"
            elif model.model_name.startswith("sequence_labeling"):
                evaluate_task = "sequence_labeling"

        if evaluate_task == "text_classify":
            return self.evaluate_text_classify(model, valid_loader)
        elif evaluate_task == "language_modeling":
            return self.evaluate_language_modeling(model, valid_loader)
        elif evaluate_task == "sequence_labeling":
            return self.evaluate_sequence_labeling(model, valid_loader)
        else:
            return self.evaluate_none_task(model, teacher_model, valid_loader)

    def evaluate_text_classify(self, model, valid_loader):
        total_loss = 0
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        logits_list = list()
        y_trues = list()

        total_spent_time = 0.0
        for _step, batch in enumerate(valid_loader):
            batch = {key: val.cuda() if isinstance(val, torch.Tensor) else val
                     for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                student_outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert "logits" in student_outputs and "label_ids" in batch
            logits, label_ids = student_outputs["logits"], batch["label_ids"]

            y_trues.extend(label_ids.tolist())
            logits_list.extend(logits.tolist())
            hit_num += torch.sum(torch.argmax(logits, dim=-1) == label_ids).item()
            total_num += label_ids.shape[0]

            if len(logits.shape) == 1 or logits.shape[-1] == 1:
                tmp_loss = losses.mse_loss(logits, label_ids)
            elif len(logits.shape) == 2:
                tmp_loss = losses.cross_entropy(logits, label_ids)
            else:
                raise RuntimeError

            total_loss += tmp_loss.mean().item()
            total_steps += 1
            total_samples += valid_loader.batch_size
            if (_step + 1) % 100 == 0:
                logger.info(
                    "Eval: %d/%d steps finished" % (_step + 1, len(valid_loader.dataset) // valid_loader.batch_size))

        logger.info("Inference time = {:.2f}s, [{:.4f} ms / sample] ".format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info("Eval loss: {}".format(eval_loss))

        logits_list = np.array(logits_list)
        eval_outputs = list()
        for metric in self.metrics:
            if metric.endswith("accuracy"):
                acc = hit_num / total_num
                logger.info("Accuracy: {}".format(acc))
                eval_outputs.append(("accuracy", acc))
            elif metric == "f1":
                f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1))
                logger.info("F1: {}".format(f1))
                eval_outputs.append(("f1", f1))
            elif metric == "macro-f1":
                f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1), average="macro")
                logger.info("Macro F1: {}".format(f1))
                eval_outputs.append(("macro-f1", f1))
            elif metric == "micro-f1":
                f1 = f1_score(y_trues, np.argmax(logits_list, axis=-1), average="micro")
                logger.info("Micro F1: {}".format(f1))
                eval_outputs.append(("micro-f1", f1))
            elif metric == "auc":
                auc = roc_auc_score(y_trues, np.argmax(logits_list, axis=-1))
                logger.info("AUC: {}".format(auc))
                eval_outputs.append(("auc", auc))
            elif metric == "matthews_corrcoef":
                mcc = matthews_corrcoef(y_trues, np.argmax(logits_list, axis=-1))
                logger.info("Matthews Corrcoef: {}".format(mcc))
                eval_outputs.append(("matthews_corrcoef", mcc))
            elif metric == "pearson_and_spearman":
                preds = logits_list[:, 0]
                pearson_corr = pearsonr(preds, y_trues)[0]
                spearman_corr = spearmanr(preds, y_trues)[0]
                logger.info("Peasrson: {}".format(pearson_corr))
                logger.info("Spearmanr: {}".format(spearman_corr))
                corr = (pearson_corr + spearman_corr) / 2.0
                logger.info("Peasrson_and_spearmanr: {}".format(corr))
                eval_outputs.append(("pearson_and_spearman", corr))
            elif metric == "classification_report":
                logger.info("\n{}".format(
                    classification_report(y_trues, np.argmax(logits_list, axis=-1), digits=4)))
            elif "last_layer_mse" in self.metrics:
                logger.info("Last layer MSE: {}".format(eval_loss))
                eval_outputs.append(("last_layer_mse", -eval_loss))
            else:
                raise NotImplementedError("Metric %s not implemented" % metric)
        return eval_outputs

    def evaluate_language_modeling(self, model, valid_loader):
        total_loss = 0
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        total_spent_time = 0.0
        for _step, batch in enumerate(valid_loader):
            batch = {key: val.cuda() if isinstance(val, torch.Tensor) else val
                     for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                student_outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert "logits" in student_outputs and "label_ids" in batch
            logits, label_ids = student_outputs["logits"], batch["label_ids"]

            for b in range(label_ids.shape[0]):
                _logits = logits[b]
                _label_ids = label_ids[b]
                mask_span_indices = batch["mask_span_indices"][b]
                for span_indices in mask_span_indices:
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
            total_samples += valid_loader.batch_size
            if (_step + 1) % 100 == 0:
                logger.info(
                    "Eval: %d/%d steps finished" % (_step + 1, len(valid_loader.dataset) // valid_loader.batch_size))

        logger.info("Inference time = {:.2f}s, [{:.4f} ms / sample] ".format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info("Eval loss: {}".format(eval_loss))
        acc = hit_num / total_num
        logger.info("Accuracy: {}".format(acc))
        eval_outputs = [("accuracy", acc)]

        return eval_outputs

    def evaluate_sequence_labeling(self, model, valid_loader):
        def predict_sequence_labeling(raw_preds, raw_label_ids, label_enumerate_values, tok_to_orig_indexes):
            new_preds = list()
            new_labels = list()
            idx_label_map = dict({idx: value for idx, value in enumerate(label_enumerate_values)})
            for idx, (raw_pred, tok_to_orig_index) in enumerate(zip(raw_preds, tok_to_orig_indexes)):
                raw_label = raw_label_ids[idx]
                final_pred = list()
                final_label = list()
                prev_token_idx = -1
                for k in range(min(len(raw_pred), len(tok_to_orig_index))):
                    token_pred = raw_pred[k]
                    token_label = raw_label[k]
                    token_orig_idx = tok_to_orig_index[k]
                    if token_orig_idx == -100:
                        continue
                    if token_orig_idx == prev_token_idx:
                        continue
                    final_pred.append(idx_label_map[token_pred])
                    final_label.append(idx_label_map[token_label])
                    prev_token_idx = token_orig_idx
                raw_sequence_length = max(tok_to_orig_index) + 1
                while len(final_pred) < raw_sequence_length:
                    final_pred.append(idx_label_map[len(idx_label_map) - 1])
                new_preds.extend(final_pred + ["O"])
                new_labels.extend(final_label + ["O"])
            return new_preds, new_labels

        total_loss = 0
        total_steps = 0
        total_samples = 0
        true_seqs = list()
        pred_seqs = list()

        total_spent_time = 0.0
        for _step, batch in enumerate(valid_loader):
            batch = {key: val.cuda() if isinstance(val, torch.Tensor) else val
                     for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                student_outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert "logits" in student_outputs and "label_ids" in batch
            logits, label_ids = student_outputs["logits"], batch["label_ids"]

            raw_preds = torch.argmax(logits, dim=-1).tolist()
            raw_label_ids = label_ids.tolist()
            new_preds, new_labels = predict_sequence_labeling(raw_preds,
                                                              raw_label_ids,
                                                              valid_loader.dataset.label_enumerate_values,
                                                              batch["tok_to_orig_index"])
            pred_seqs.extend(new_preds)
            true_seqs.extend(new_labels)
            logits = logits.view(-1, logits.size(-1))
            label_ids = label_ids.view(-1)

            tmp_loss = losses.cross_entropy(logits, label_ids)

            total_loss += tmp_loss.mean().item()
            total_steps += 1
            total_samples += valid_loader.batch_size
            if (_step + 1) % 100 == 0:
                logger.info(
                    "Eval: %d/%d steps finished" % (_step + 1, len(valid_loader.dataset) // valid_loader.batch_size))

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

    def evaluate_none_task(self, model, teacher_model, valid_loader):
        if teacher_model is None:
            return [('no-metric', float("-inf"))]

        total_loss = 0
        total_steps = 0
        total_samples = 0

        total_spent_time = 0.0
        for _step, batch in enumerate(valid_loader):
            batch = {key: val.cuda() if isinstance(val, torch.Tensor) else val
                     for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                student_outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            with torch.no_grad():
                teacher_outputs = teacher_model(batch)
                student_hidn = student_outputs["hidden"][-1]
                teacher_hidn = teacher_outputs["hidden"][-1]
                tmp_loss = losses.mse_loss(student_hidn, teacher_hidn)

            total_loss += tmp_loss.mean().item()
            total_steps += 1
            total_samples += valid_loader.batch_size
            if (_step + 1) % 100 == 0:
                logger.info(
                    "Eval: %d/%d steps finished" % (_step + 1, len(valid_loader.dataset) // valid_loader.batch_size))

        logger.info("Inference time = {:.2f}s, [{:.4f} ms / sample] ".format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info("Eval loss: {}".format(eval_loss))

        return [("last_layer_mse", -eval_loss)]