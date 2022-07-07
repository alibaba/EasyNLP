# coding=utf-8
# Copyright 2018 Mindspore team, Alibaba PAI team.
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
""" Wukong model configuration"""

import copy
import os
from typing import Union
import json
from ...configuration_utils import PretrainedConfig
from ...utils import logging

class WukongConfig(PretrainedConfig):
    model_type = "wukong"
    def __init__(self,config_obj):
        super().__init__()
        self.data=config_obj
    
    def to_json_string(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.data)
        return json.dumps(output)
