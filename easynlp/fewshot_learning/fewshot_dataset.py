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
import random
import traceback

import torch

from easynlp.appzoo.dataset import BaseDataset
from easynlp.modelzoo import AutoTokenizer
from easynlp.utils import io


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class FewshotBaseDataset(BaseDataset):
    """This is a dataset class for supporting fewshot learning.

    Args:
        pretrained_model_name_or_path:
            The name of the path of th pretrained model for fewshot classification.
        data_file:
            The input data file.
        max_seq_length:
            The maximum sequence length for the transformer model.
        first_sequence:
            The name of the first sequence in the input schema.
        input_schema:
            The schema of the input data file.
        user_defined_parameters:
            The dict of user defined parameters for fewshot classification.
        label_name:
            The name of the label name in the input schema.
        second_sequence:
            The name of the second sequence in the input schema.
        label_enumerate_values:
            The string of all label values, seperated by comma.
    """
    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 first_sequence,
                 input_schema=None,
                 user_defined_parameters=None,
                 label_name=None,
                 second_sequence=None,
                 label_enumerate_values=None,
                 **kwargs):
        super(FewshotBaseDataset, self).__init__(data_file,
                                                 input_schema=input_schema,
                                                 output_format='dict',
                                                 **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)
        self.max_seq_length = max_seq_length
        if label_enumerate_values is None:
            self._label_enumerate_values = '0,1'.split(',')
        else:
            if io.exists(label_enumerate_values):
                with io.open(label_enumerate_values) as f:
                    self._label_enumerate_values = [line.strip() for line in f]
            else:
                self._label_enumerate_values = label_enumerate_values.split(
                    ',')
        assert first_sequence in self.column_names, \
            'Column name %s needs to be included in columns' % first_sequence
        self.first_sequence = first_sequence

        if second_sequence:
            assert second_sequence in self.column_names, \
                'Column name %s needs to be included in columns' % second_sequence
            self.second_sequence = second_sequence
        else:
            self.second_sequence = None

        if label_name:
            assert label_name in self.column_names, \
                'Column name %s needs to be included in columns' % label_name
            self.label_name = label_name
        else:
            self.label_name = 'label'
            self.vocab_size = len(self.tokenizer.vocab)
        self.pad_idx = self.tokenizer.pad_token_id
        self.mask_idx = self.tokenizer.mask_token_id

        # for model random initialization
        try:
            user_defined_parameters_dict = user_defined_parameters.get(
                'app_parameters')
        except KeyError:
            traceback.print_exc()
            exit(-1)
        pattern = user_defined_parameters_dict.get('pattern')
        assert pattern is not None, 'You must define the pattern for PET learning'
        pattern_list = pattern.split(',')
        assert self.first_sequence in pattern_list and (
                    not self.second_sequence or self.second_sequence in pattern_list) \
            , 'All text columns should be included in the pattern'

        # add special token for pseudo tokens
        cnt = 0
        for i in range(len(pattern_list)):
            if pattern_list[i] == '<pseudo>':
                pattern_list[i] = '<pseudo-%d>' % cnt
                cnt += 1
        if cnt > 0:
            self.tokenizer.add_tokens(['<pseudo-%d>' % i for i in range(cnt)])
        self.pattern = [
            self.tokenizer.tokenize(s) if s not in (self.first_sequence,
                                                    self.second_sequence,
                                                    self.label_name) else s
            for s in pattern_list
        ]

        label_desc = user_defined_parameters_dict.get('label_desc')
        if not label_desc:
            print(
                'Using Contrastive Few shot Learner, using random label words only as place-holders'
            )
            label_desc = [s[0] for s in label_enumerate_values.split(',')]
        else:
            label_desc = label_desc.split(',')
        self.masked_length = len(self.tokenizer.tokenize(label_desc[0]))
        self.label_map = dict(
            zip(label_enumerate_values.split(','), label_desc))
        self.num_extra_tokens = sum(
            [len(s) if isinstance(s, list) else 0
             for s in self.pattern]) + self.masked_length

    @property
    def eval_metrics(self):
        return ('mlm_accuracy', )

    @property
    def label_enumerate_values(self):
        return self._label_enumerate_values

    def convert_single_row_to_example(self, row):
        """Converting the examples into the dict of values."""
        text_a = row[self.first_sequence]
        text_b = row[self.second_sequence] if self.second_sequence else None
        label = row[self.label_name] if self.label_name else None
        tokens_a = self.tokenizer.tokenize(text_a)
        max_seq_length = self.max_seq_length
        max_seq_length -= self.num_extra_tokens
        tokens_b = None
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        if label is None:
            # Prediction mode
            label = self.tokenizer.mask_token * self.masked_length
        elif self.label_map:
            label = self.label_map[label]
        elif label is not None:
            assert isinstance(label, str), type(label)
        else:
            raise ValueError('Undefined situation for label transformation')
        label_tokens = self.tokenizer.tokenize(label)
        assert len(
            label_tokens
        ) == self.masked_length, 'label length %d should be equal to the mask length %d' % (
            len(label_tokens), self.masked_length)
        tokens = [self.tokenizer.cls_token]
        label_position = None
        for p in self.pattern:
            if p == self.first_sequence:
                tokens += tokens_a
            elif p == self.second_sequence:
                tokens += (tokens_b if tokens_b else [])
            elif p == self.label_name:
                label_position = len(tokens)
                tokens += [
                    self.tokenizer.mask_token,
                ] * self.masked_length
            elif isinstance(p, list):
                tokens += p
            else:
                raise ValueError('Unexpected pattern---' + p)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label_tokens_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
        length = len(input_ids)
        attention_mask = [1] * length
        token_type_ids = [0] * length
        mask_labels = [-100] * length
        mask_span_indices = []
        for i in range(self.masked_length):
            mask_labels[label_position + i] = label_tokens_ids[i]
            mask_span_indices.append([label_position + i])
        max_seq_length += self.num_extra_tokens
        # token padding
        input_ids += [self.pad_idx] * (max_seq_length - length)
        attention_mask += [self.pad_idx] * (max_seq_length - length)
        token_type_ids += [0] * (max_seq_length - length)
        mask_labels += [-100] * (max_seq_length - length)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_ids': mask_labels,
            'mask_span_indices': mask_span_indices
        }

    def batch_fn(self, features):
        """Divide examples into batches."""
        return {
            k: torch.tensor([dic[k] for dic in features], dtype=torch.long)
            for k in features[0]
        }
