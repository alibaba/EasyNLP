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

import numpy as np
import torch
from torch.utils.data import DataLoader

from easynlp.core.evaluator import Evaluator
from easynlp.fewshot_learning.fewshot_dataset import FewshotBaseDataset
from easynlp.utils import losses
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils.logger import logger


class PromptEvaluator(Evaluator):
    """An evaluator class for supporting fewshot learning (PET and P-tuning)."""
    def __init__(self, valid_dataset, **kwargs):
        super(PromptEvaluator, self).__init__(valid_dataset, **kwargs)

        self.metrics = ['mlm_accuracy']

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        total_spent_time = 0.0

        def str_to_ids(s, tokenizer):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

        label_candidates_ids = [
            str_to_ids(v, self.valid_loader.dataset.tokenizer)
            for v in self.valid_loader.dataset.label_map.values()
        ]

        for _step, batch in enumerate(self.valid_loader):
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop('label_ids')
                mask_span_indices = batch.pop('mask_span_indices')
                outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            logits = outputs['logits']

            for b in range(label_ids.shape[0]):  #bz
                _logits = logits[b]
                _label_ids = label_ids[b]
                indices = mask_span_indices[b]
                y_pred = torch.nn.functional.log_softmax(_logits)
                label = list()
                label_prob = 0.
                for idx, span_indices in enumerate(indices):
                    span_idx = span_indices[0]
                    label.append(_label_ids[span_idx].item())
                    label_prob += y_pred[span_idx, _label_ids[span_idx].item()]

                pred_correct = True
                for l in label_candidates_ids:
                    if tuple(l) == tuple(label):
                        continue
                    pred_prob = 0.
                    for l_ids, span_indices in zip(l, indices):
                        span_idx = span_indices[0]
                        pred_prob += y_pred[span_idx, l_ids]
                    if pred_prob > label_prob:
                        pred_correct = False
                        break
                if pred_correct:
                    hit_num += 1
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
                logger.info('Eval: %d/%d steps finished' %
                            (_step + 1, len(self.valid_loader.dataset) //
                             self.valid_loader.batch_size))

        logger.info('Inference time = {:.2f}s, [{:.4f} ms / sample] '.format(
            total_spent_time, total_spent_time * 1000 / total_samples))

        eval_loss = total_loss / total_steps
        logger.info('Eval loss: {}'.format(eval_loss))
        acc = hit_num / total_num
        logger.info('Accuracy: {}'.format(acc))
        eval_outputs = [('accuracy', acc)]

        return eval_outputs


class CPTEvaluator(Evaluator):
    """An evaluator class for supporting CPT fewshot learning."""
    def __init__(self, valid_dataset, *args, **kwargs):
        super(CPTEvaluator, self).__init__(valid_dataset, *args, **kwargs)
        self.metrics = ['mlm_accuracy']

        anchor_args = kwargs.pop('few_shot_anchor_args', None)
        anchor_dataset = FewshotBaseDataset(
            data_file=anchor_args.tables.split(',')[0],
            pretrained_model_name_or_path=anchor_args.
            pretrained_model_name_or_path,
            max_seq_length=anchor_args.sequence_length,
            first_sequence=anchor_args.first_sequence,
            input_schema=anchor_args.input_schema,
            second_sequence=anchor_args.second_sequence,
            label_name=anchor_args.label_name,
            label_enumerate_values=anchor_args.label_enumerate_values,
            user_defined_parameters=parse_user_defined_parameters(
                anchor_args.user_defined_parameters),
            is_training=True,
            *args,
            **kwargs)

        assert anchor_dataset is not None, 'anchor_dataset should be included for this task.'
        eval_batch_size = kwargs.get('eval_batch_size', 32)
        self.anchor_dataloader = DataLoader(anchor_dataset,
                                            batch_size=eval_batch_size,
                                            shuffle=False,
                                            collate_fn=anchor_dataset.batch_fn)

    def evaluate(self, model):
        model.eval()
        total_steps = 0
        total_samples = 0
        hit_num = 0
        total_num = 0

        total_spent_time = 0.0
        anchor_list = []
        anchor_labels = []
        print('Calculating anchor features')
        for _, batch in enumerate(self.anchor_dataloader):
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            with torch.no_grad():
                label_ids = batch.pop('label_ids')
                outputs = model(batch)
                labels = label_ids[label_ids > 0].detach().cpu().numpy()
                features = torch.nn.functional.normalize(
                    outputs['features'][label_ids > 0]).detach().cpu().numpy()
                anchor_list.append(features)
                anchor_labels.append(labels)
        anchor_feature = np.concatenate(anchor_list)
        anchor_labels = np.concatenate(anchor_labels)

        def str_to_ids(s, tokenizer):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

        label_candidates_ids = [
            str_to_ids(v, self.valid_loader.dataset.tokenizer)
            for v in self.valid_loader.dataset.label_map.values()
        ]
        for _step, batch in enumerate(self.valid_loader):
            batch = {
                key: val.cuda() if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop('label_ids')
                mask_span_indices = batch.pop('mask_span_indices')
                outputs = model(batch)
                labels = label_ids[label_ids > 0].detach().cpu().numpy()
                features = torch.nn.functional.normalize(
                    outputs['features'][label_ids > 0]).detach().cpu().numpy()
                dist = np.dot(features, anchor_feature.T)
                pred = anchor_labels[np.argmax(dist, -1)]
                total_num += features.shape[0]
                hit_num += np.sum(pred == labels)

            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time
            total_steps += 1
            if (_step + 1) % 100 == 0:
                logger.info('Eval: %d/%d steps finished' %
                            (_step + 1, len(self.valid_loader.dataset) //
                             self.valid_loader.batch_size))
            total_samples += self.valid_loader.batch_size

        logger.info('Inference time = {:.2f}s, [{:.4f} ms / sample] '.format(
            total_spent_time, total_spent_time * 1000 / total_samples))
        acc = hit_num / total_num
        logger.info('Accuracy: {}'.format(acc))
        eval_outputs = [('accuracy', acc)]

        return eval_outputs
