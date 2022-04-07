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

from ..application import Application
from ...modelzoo import AutoModel, AutoConfig


class FeatureVectorization(Application):

    def __init__(self, pretrained_model_name_or_path=None, **kwargs):
        super().__init__()
        if kwargs.get('from_config'):
            # for evaluation and prediction
            self.config = kwargs.get('from_config')
            self.backbone = AutoModel.from_config(self.config)
        elif kwargs.get('user_defined_parameters') is not None and \
            "model_parameters" in kwargs.get('user_defined_parameters'):
            # for model random initialization
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            user_defined_parameters = kwargs.get('user_defined_parameters')
            user_defined_parameters_dict = literal_eval(user_defined_parameters)
            self.config.update(user_defined_parameters_dict['model_parameters'])
            self.backbone = AutoModel.from_config(self.config)
        else:
            # for pretrained model, initialize from the pretrained model
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)
    
    def forward(self, inputs):
        outputs = self.backbone(**inputs, output_hidden_states=True, output_attentions=True)
        sequence_output = outputs['last_hidden_state']
        pooler_output = outputs['pooler_output']
        # sequence output: [32, 128, 768]
        first_token_output = sequence_output[:, 0, :]
        all_hidden_outputs = sequence_output[:, 1:, :]
        return {
            "pooler_output": pooler_output,
            "first_token_output": first_token_output,
            "all_hidden_outputs": all_hidden_outputs
        }
