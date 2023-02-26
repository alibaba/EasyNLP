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

import io
import os

import torch

from ..appzoo.dataset import BaseDataset
from ..modelzoo import AutoTokenizer


class MetakdSentiClassificationDataset(BaseDataset):
    """ A dataset class for supporting metakd knowledge distillation. This class is base on :class:`BaseDataset`
    additional args:
        genre: the domain of dataset, choosing all domains data when genre is `all`.

        domain_label: a list of domain in the dataset, the domain list of senti datasets is default value.

    """
    def __init__(
            self,
            pretrained_model_name_or_path,
            data_file,
            max_seq_length,
            input_schema,
            first_sequence,
            label_name=None,
            second_sequence=None,
            label_enumerate_values=None,
            # multi_label=False,
            *args,
            **kwargs):
        """We overwrite the function `readlines_from_file` of the BaseClass, so need to add some attributes before the ParentClass is initialized."""
        if 'genre' in kwargs:
            self.genre = kwargs['genre']
        else:
            self.genre = None
        if 'domain_label' in kwargs:
            self.domain_list = kwargs['domain_label'].split(',')
        else:
            self.domain_list = ['books', 'dvd', 'electronics', 'kitchen']

        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format='dict',
                         *args,
                         **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=False)
        self.max_seq_length = max_seq_length

        # self.multi_label = multi_label
        if label_enumerate_values is None:
            self._label_enumerate_values = '0,1'.split(',')
        else:
            if os.path.exists(label_enumerate_values):
                with io.open(label_enumerate_values) as f:
                    self._label_enumerate_values = [line.strip() for line in f]
            else:
                self._label_enumerate_values = label_enumerate_values.split(
                    ',')
        self.max_num_labels = len(self._label_enumerate_values)
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
            self.label_name = None

        self.label_map = dict({
            value: idx
            for idx, value in enumerate(self.label_enumerate_values)
        })
        self.domain_idx_mapping = {
            domain: idx
            for idx, domain in enumerate(self.domain_list)
        }

    @property
    def label_enumerate_values(self):
        """Returns the label enumerate values."""
        return self._label_enumerate_values

    def readlines_from_file(self, data_file, skip_first_line=None):
        i = 0
        if skip_first_line is None:
            skip_first_line = self.skip_first_line
        with io.open(data_file) as f:
            if skip_first_line:
                f.readline()
            data_rows = f.readlines()
        if self.genre:
            if self.genre in self.domain_list:
                return [
                    row for row in data_rows
                    if row.strip().split('\t')[4] == self.genre
                ]
        else:
            return data_rows

    def convert_single_row_to_example(self, row):
        """
            Convert sample token to indices.Overrides the methods of the parent class as required.
            Args:
                row: contains sequence and label.

                text_a: the first sequence in row.

                text_b: the second sequence in row if self.second_sequence is true.

                label: label token if self.label_name is true.

                domain: the domain of sequence in row.

                weight: the weight calculated after pre-processing.

            Returns: sing example
                encoding: an example contains token indices.
                A dict additional contains:
                domain_id: the domain id of sequence through mapped.

                label_ids: the label id of sequence through mapped.

                sample_weights: same to the weight.

        """
        text_a = row[self.first_sequence]
        text_b = row[self.second_sequence] if self.second_sequence else None
        label = row[self.label_name] if self.label_name else None

        # guid = row[self.guid]
        try:
            domain_id = self.domain_idx_mapping[row['domain']]
        except:
            raise RuntimeError(
                "Can't load data in dataset files! Checking input_schema.")
        # domain = row[self.domain]

        encoding = self.tokenizer(text_a,
                                  text_b,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_length)

        encoding['label_ids'] = int(self.label_map[label])
        encoding['domain_ids'] = int(domain_id)

        try:
            encoding['sample_weights'] = float(row['weight'])
        except:
            raise RuntimeError("Can't load weight in dataset files, \
                you might forget to preprocess the init dataset.")

        return encoding

    def batch_fn(self, features):
        """Divide examples into batches."""
        return {
            k: torch.tensor([dic[k] for dic in features])
            for k in features[0]
        }
