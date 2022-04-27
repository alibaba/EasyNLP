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

import json

with open('./data/meta_info.json', 'r') as file:
    meta_data = eval(json.load(file))['data']
import os
for index, task in enumerate(meta_data):
    base_dir = './MeLL_pytorch/data/'
    if index >=40:
        taskKey = task['taskKey']
        base_dir += taskKey
        os.makedirs(base_dir)
        with open(base_dir+'/'+'lifelong_task.json', 'w') as file:
            json.dump(str({"data":task}), file)