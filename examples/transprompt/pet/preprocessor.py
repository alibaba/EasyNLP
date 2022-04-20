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

# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:10 pm
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @Github  : https://github.com/alibaba/EasyTransfer, https://github.com/wjn1996

from abc import ABC, abstractmethod
from pet.utils import InputFeatures, InputExample
from data_utils.task_pvps import PVP, PVPS
from data_utils.utils import task_to_id
from pet.config import WrapperConfig

class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """

    def __init__(self, wrapper, args: WrapperConfig, task_name, pattern_id: int = 0):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        :param task_name: the name of the task
        :param pattern_id: the id of the PVP to be used
        :param verbalizer_file: path to a file containing a verbalizer that overrides the default verbalizer
        """
        self.wrapper = wrapper
        self.args = args # add by wjn
        self.pvp = PVPS[task_name](self.wrapper, self.args, pattern_id)  # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""

    def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                           **kwargs) -> InputFeatures:
        # 获得PVP（模板句子+label mapping）
        input_ids, token_type_ids, block_flag = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length) # wordid序列+padding
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        block_flag = block_flag + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length
        assert len(block_flag) == self.wrapper.config.max_seq_length

        example_label = example.label
        example_task = example.task # add by wjn 表示当前样本所属的task
        # add by wjn 只有当数字型的label（0,1），可能真实标签是字符串（'0', '1'），因此需要进行转换判断
        if example_label not in self.label_map.keys():
            if type(example_label) == int:
                example_label = str(example_label)
            elif type(example_label) == str:
                example_label = int(example_label)


        label = self.label_map[example_label] if example.label is not None else -100
        task = task_to_id[example_task] # add by wjn 表示当前task对应group内的编号
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(guid=int(example.guid.split('-')[1]),
                             input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             task=task,
                             label=label, # [0,1,..]
                             mlm_labels=mlm_labels, # [-1, -1, .., -1, 1, -1, -1, ...]
                             logits=logits,
                             idx=example.idx,
                             block_flag=block_flag)
