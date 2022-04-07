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

from torch.utils.data import DataLoader


class Evaluator(object):
    def __init__(self, valid_dataset, **kwargs):
        eval_batch_size = kwargs.get('eval_batch_size', 32)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=eval_batch_size,
                                       shuffle=False,
                                       collate_fn=valid_dataset.batch_fn)

        self.best_valid_score = float('-inf')

    def evaluate(self, model):
        raise NotImplementedError

    @property
    def eval_metrics(self):
        raise NotImplementedError
