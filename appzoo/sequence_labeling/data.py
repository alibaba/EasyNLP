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

from ..dataset import BaseDataset
from ...modelzoo import AutoTokenizer


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None, guid=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.guid = guid


class LabelingFeatures(object):
    """A single set of features of data for sequence labeling."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 all_tokens,
                 label_ids,
                 tok_to_orig_index,
                 seq_length=None,
                 guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.all_tokens = all_tokens
        self.seq_length = seq_length
        self.label_ids = label_ids
        self.tok_to_orig_index = tok_to_orig_index
        self.guid = guid


def bert_labeling_convert_example_to_feature(example, tokenizer, max_seq_length, label_map=None):
    """ Convert `InputExample` into `InputFeature` For sequence labeling task

        Args:
            example (`InputExample`): an input example
            tokenizer (`BertTokenizer`): BERT Tokenizer
            max_seq_length (`int`): Maximum sequence length while truncating
            label_map (`dict`): a map from label_value --> label_idx,
                                "regression" task if it is None else "classification"
        Returns:
            feature (`InputFeatures`): an input feature
    """

    content_tokens = example.text_a.split(" ")
    if example.label is not None:
        label_tags = example.label.split(" ")
    else:
        label_tags = None

    all_tokens = ["[CLS]"]
    all_labels = [""]
    tok_to_orig_index = [-100]
    for i, token in enumerate(content_tokens):
        sub_tokens = tokenizer.tokenize(token)
        if not sub_tokens:
            sub_tokens = ["[UNK]"]
        all_tokens.extend(sub_tokens)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        if label_tags is None:
            all_labels.extend(["" for _ in range(len(sub_tokens))])
        else:
            all_labels.extend([label_tags[i] for _ in range(len(sub_tokens))])
    all_tokens = all_tokens[:max_seq_length - 1]
    all_labels = all_labels[:max_seq_length - 1]
    all_tokens.append("[SEP]")
    all_labels.append("")
    tok_to_orig_index.append(-100)

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    label_ids = [label_map[label] if label else -100 for label in all_labels]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(-100)

    feature = LabelingFeatures(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_ids=label_ids,
                               all_tokens=all_tokens,
                               seq_length=max_seq_length,
                               tok_to_orig_index=tok_to_orig_index,
                               guid=example.guid)

    return feature


class SequenceLabelingDataset(BaseDataset):
    """ Sequence Labeling Dataset

    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file.
        max_seq_length: max sequence length of each input instance.
        first_sequence: input sequence.
        label_name: label column name.
        label_enumerate_values: the list of label values.
    """
    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                #  input_schema,
                 first_sequence,
                 label_name=None,
                 label_enumerate_values=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         output_format="dict",
                         *args,
                         **kwargs)

        # assert ".easynlp/modelzoo/" in pretrained_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.max_seq_length = max_seq_length

        if label_enumerate_values is None:
            self._label_enumerate_values = "0,1".split(",")
        else:
            self._label_enumerate_values = label_enumerate_values.split(",")

        assert first_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % first_sequence
        self.first_sequence = first_sequence

        if label_name:
            assert label_name in self.column_names, \
                "Column name %s needs to be included in columns" % label_name
            self.label_name = label_name
        else:
            self.label_name = None

        self.label_map = dict({value: idx for idx, value in enumerate(self.label_enumerate_values)})

    @property
    def label_enumerate_values(self):
        return self._label_enumerate_values

    def convert_single_row_to_example(self, row):
        text_a = row[self.first_sequence]
        text_b = None
        label = row[self.label_name] if self.label_name else None
        example = InputExample(text_a=text_a, text_b=text_b, label=label)
        return bert_labeling_convert_example_to_feature(example, self.tokenizer,
                                                        self.max_seq_length, self.label_map)

    def batch_fn(self, features):
        inputs = {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
            "token_type_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
            "label_ids": torch.tensor([f.label_ids for f in features], dtype=torch.long),
            "tok_to_orig_index": [f.tok_to_orig_index for f in features]
        }
        return inputs
