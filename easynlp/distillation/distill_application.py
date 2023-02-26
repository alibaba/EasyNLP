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

from ..appzoo.application import Application
from ..utils import losses


class DistillatoryBaseApplication(Application):
    """This is the application class for for supporting knowledge distillation."""
    def compute_loss(self, forward_outputs, label_ids, teacher_logits,
                     **kwargs):
        """Computing the knowledge distillation loss based on teacher logits.

        Args:
            forward_outputs:
                the dict of the output tensors of the student model
            label_ids:
                the true label ids
            teacher_logits:
                the tensor of teacher logits

        Returns: the dict of output tensors containing the loss
        """

        kd_type = kwargs.pop('type')
        logits = forward_outputs['logits']
        if kd_type == 'vanilla_kd':
            loss = losses.vanilla_loss(logits, teacher_logits, label_ids,
                                       **kwargs)
        else:
            raise NotImplementedError(
                f'KD type {kd_type} is not available yet, please use '
                'supported KD methods.')
        return {'loss': loss}
