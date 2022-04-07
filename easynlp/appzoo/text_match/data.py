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
import numpy as np
from ...distillation.distill_dataset import DistillatoryBaseDataset
from ..dataset import BaseDataset
from ...fewshot_learning.fewshot_dataset import FewshotBaseDataset
from ...modelzoo import AutoTokenizer
from torch.utils.data import Dataset
from ...utils import io
from ...utils import parse_row_by_schema


class SingleTowerDataset(BaseDataset):

    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 label_name=None,
                 second_sequence=None,
                 label_enumerate_values=None,
                 multi_label=False,
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
        self.multi_label = multi_label

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
            return encoding
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


class TwoTowerDataset(BaseDataset):

    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 label_name=None,
                 second_sequence=None,
                 label_enumerate_values=None,
                 multi_label=False,
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
        self.multi_label = multi_label

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

        encoding_a = self.tokenizer(text_a,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_seq_length)

        encoding_b = self.tokenizer(text_b,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_seq_length)
        encoding = {}
        if not self.multi_label:
            encoding['label_ids'] = self.label_map[label]
            encoding['input_ids_a'] = encoding_a.pop('input_ids')
            encoding['token_type_ids_a'] = encoding_a.pop('token_type_ids')
            encoding['attention_mask_a'] = encoding_a.pop('attention_mask')
            encoding['input_ids_b'] = encoding_b.pop('input_ids')
            encoding['token_type_ids_b'] = encoding_b.pop('token_type_ids')
            encoding['attention_mask_b'] = encoding_b.pop('attention_mask')
            return encoding
        else:
            label_id = [self.label_map[x] for x in label.split(",") if x]
            new_label_id = [0] * self.max_num_labels
            for idx in label_id:
                new_label_id[idx] = 1
            encoding['label_ids'] = new_label_id
            encoding['input_ids_a'] = encoding_a.pop('input_ids')
            encoding['token_type_ids_a'] = encoding_a.pop('token_type_ids')
            encoding['attention_mask_a'] = encoding_a.pop('attention_mask')
            encoding['input_ids_b'] = encoding_b.pop('input_ids')
            encoding['token_type_ids_b'] = encoding_b.pop('token_type_ids')
            encoding['attention_mask_b'] = encoding_b.pop('attention_mask')
            return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}


class SiameseDataset(Dataset):
    def __init__(self, dataset, *args, **kwargs):
        is_training = kwargs.get('is_training', False)
        self.cls_dataset = dataset
        self.is_training = is_training
        self.input_schema = dataset.input_schema
        self.first_sequence = dataset.first_sequence
        self.label_enumerate_values = dataset.label_enumerate_values

        if self.is_training:
            self.train_labels = self.cls_dataset.labels
            self.train_data = self.cls_dataset.data_rows
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.cls_dataset.labels
            self.test_data = self.cls_dataset.data_rows
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}
            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.is_training:
            target = np.random.randint(0, 2)
            row, label1 = self.train_data[index], self.train_labels[index].item()
            row = parse_row_by_schema(row, self.input_schema)
            sent1 = row[self.first_sequence]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            row = self.train_data[siamese_index]
            row = parse_row_by_schema(row, self.input_schema)
            sent2 = row[self.first_sequence]

        else:
            row = self.test_data[self.test_pairs[index][0]]
            row = parse_row_by_schema(row, self.input_schema)
            sent1 = row[self.first_sequence]

            row = self.test_data[self.test_pairs[index][1]]
            row = parse_row_by_schema(row, self.input_schema)
            sent2 = row[self.first_sequence]
            target = self.test_pairs[index][2]

        encoding_a = self.cls_dataset.tokenizer(sent1, padding='max_length', truncation=True,
                                  max_length=self.cls_dataset.max_seq_length)

        encoding_b = self.cls_dataset.tokenizer(sent2, padding='max_length', truncation=True,
                                  max_length=self.cls_dataset.max_seq_length)
        encoding = {}
        encoding['label_ids'] = target
        encoding['input_ids_a'] = encoding_a.pop('input_ids')
        encoding['token_type_ids_a'] = encoding_a.pop('token_type_ids')
        encoding['attention_mask_a'] = encoding_a.pop('attention_mask')
        encoding['input_ids_b'] = encoding_b.pop('input_ids')
        encoding['token_type_ids_b'] = encoding_b.pop('token_type_ids')
        encoding['attention_mask_b'] = encoding_b.pop('attention_mask')
        return encoding

    def __len__(self):
        return len(self.cls_dataset)

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features], dtype=torch.long) for k in features[0]}


class DistillatorySingleTowerDataset(DistillatoryBaseDataset, SingleTowerDataset):
    pass

class FewshotSingleTowerTextMatchDataset(FewshotBaseDataset):
    pass