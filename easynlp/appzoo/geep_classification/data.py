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

import torch
from ...modelzoo import AutoTokenizer
from ...utils import io
from ..dataset import BaseDataset

class GEEPClassificationDataset(BaseDataset):
    """
    Classification Dataset
    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file.
        max_seq_length: max sequence length of each input instance.
        first_sequence: input text
        label_name: label column name
        second_sequence: set as None
        label_enumerate_values: a list of label values
        multi_label: set as True if perform multi-label classification, otherwise False
    """
    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 label_name=None,
                 second_sequence=None,
                 label_enumerate_values=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)

        # assert ".easynlp/modelzoo/" in pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.max_seq_length = max_seq_length
        user_defined_parameters = kwargs.get('user_defined_parameters', {})
        self.multi_label = user_defined_parameters.get('app_parameters', {}).get('multi_label', False)
        if label_enumerate_values is None:
            self._label_enumerate_values = "0,1".split(",")
        else:
            if io.exists(label_enumerate_values):
                with io.open(label_enumerate_values) as f:
                    self._label_enumerate_values = [line.strip() for line in f]
            else:
                self._label_enumerate_values = label_enumerate_values.split(",")
        self.max_num_labels = len(self._label_enumerate_values)
        assert first_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % first_sequence
        self.first_sequence = first_sequence
        if second_sequence:
            assert second_sequence in self.column_names, \
                "Column name %s needs to be included in columns" % second_sequence
            self.second_sequence = second_sequence
        else:
            self.second_sequence = None
        if label_name:
            assert label_name in self.column_names, \
                "Column name %s needs to be included in columns" % label_name
            self.label_name = label_name
        else:
            self.label_name = None
        self.label_map = dict({value: idx for idx, value in enumerate(self.label_enumerate_values)})

    @property
    def label_enumerate_values(self):
        """
            Returns the label enumerate values.
        """
        return self._label_enumerate_values

    def convert_single_row_to_example(self, row):
        """Convert sample token to indices.
            Args:
                row: contains sequence and label.
                text_a: the first sequence in row.
                text_b: the second sequence in row if self.second_sequence is true.
                label: label token if self.label_name is true.
            Returns: sing example
                encoding: an example contains token indices.
        """
        text_a = row[self.first_sequence]
        text_b = row[self.second_sequence] if self.second_sequence else None
        label = row[self.label_name] if self.label_name else None
        encoding = self.tokenizer(text_a,
                                  text_b,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_length)
        if not self.multi_label:
            encoding['label_ids'] = self.label_map[label]
        else:
            label_id = [self.label_map[x] for x in label.split(",") if x]
            new_label_id = [0] * self.max_num_labels
            for idx in label_id:
                new_label_id[idx] = 1
            encoding['label_ids'] = new_label_id
        return encoding
    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}