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

from ...core.evaluator import Evaluator
from tqdm import tqdm
import torch
import time
import math
import numpy as np

class OpenDomainDialogueEvaluator(Evaluator):

    def __init__(self, valid_dataset, **kwargs):
        super().__init__(valid_dataset, **kwargs)

    def evaluate(self, model):
        model.eval()
        total_loss = 0
        total_tokens = 0
        total_acc = 0
        total_samples = 0
        total_em = 0
        total_ppl = 0
        eval_outputs = list()

        total_spent_time = 0.0
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        for _step, batch in enumerate(tqdm(self.valid_loader)):
            try:
                batch = {
                    key: val.to(device) if isinstance(val, torch.Tensor) else val
                    for key, val in batch.items()
                }
            except RuntimeError:
                batch = {key: val for key, val in batch.items()}

            infer_start_time = time.time()
            with torch.no_grad():
                label_ids = batch.pop('label_ids')
                outputs = model(batch)
            infer_end_time = time.time()
            total_spent_time += infer_end_time - infer_start_time

            assert 'logits' in outputs
            logits = outputs['logits']
            preds = outputs['predictions']
            loss = model.compute_token_loss(outputs, label_ids)['loss']

            null_idx = model.backbone.NULL_IDX
            notnull = label_ids.ne(null_idx)
            target_tokens = notnull.long().sum(dim=1)
            correct = ((label_ids == preds) * notnull).sum(dim=-1)

            loss = loss.tolist()
            target_tokens = target_tokens.tolist()
            correct = correct.tolist()

            loss_avg = [l/t for l in loss for t in target_tokens]
            total_loss += sum(loss_avg)
            total_ppl += sum([math.exp(i) for i in loss_avg])
            total_acc += sum([c/t for c in correct for t in target_tokens])
            total_em += sum([correct[i]==target_tokens[i] for i in range(len(target_tokens))])
            total_samples += len(target_tokens)
        
        # cross entropy loss
        total_loss /= total_samples
        eval_outputs.append(('loss', total_loss))
        # perplexity
        total_ppl /= total_samples
        eval_outputs.append(('ppl', total_ppl))
        # token-wise accuracy
        total_acc /= total_samples
        eval_outputs.append(('token_acc', total_acc))
        # utterance-wise exact match
        total_em /= total_samples
        eval_outputs.append(('token_em', total_em))

        return eval_outputs
