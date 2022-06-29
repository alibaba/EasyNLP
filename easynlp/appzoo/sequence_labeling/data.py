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
import jieba

from ..dataset import BaseDataset
from ...modelzoo import AutoTokenizer
from ..dataset import GeneralDataset


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

class KBERTLabelingFeatures(object):
    """A single set of features of data for sequence labeling."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 all_tokens,
                 label_ids,
                 tok_to_orig_index,
                 seq_length=None,
                 guid=None,
                 visible_matrix=None,
                 position_ids=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.all_tokens = all_tokens
        self.seq_length = seq_length
        self.label_ids = label_ids
        self.tok_to_orig_index = tok_to_orig_index
        self.guid = guid
        self.visible_matrix = visible_matrix
        self.position_ids = position_ids


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



def kbert_labeling_convert_example_to_feature(example, tokenizer, max_seq_length, label_map=None,):
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

    pos_ids = [0]
    ent_visible_matrix = [[1 for _ in range(max_seq_length)] for _ in range(max_seq_length)]

    pos_id = 1
    ent_pos_id = 0
    ent_map = []  # ent_map is the log of origin_ent_id and external_ent_id
    ent_token_count = 0
    start_origin_ent = None
    start_external_ent = None

    content_tokens = example.text_a.split(" ")
    if example.label is not None:
        label_tags = example.label.split(" ")
    else:
        label_tags = None

    all_tokens = ["[CLS]"]
    all_labels = [""]
    tok_to_orig_index = [-100]

    # 记录多余的token数量，保证label 和原始对应
    ent_count = 0

    for i, token in enumerate(content_tokens):
        if token == "[ENT]":
            ent_token_count += 1
            ent_count += 1
            ent_pos_id = 0
            if ent_token_count % 3 == 1:
                start_origin_ent = len(all_tokens)
            elif ent_token_count % 3 == 2:
                start_external_ent = len(all_tokens)
            else:
                if len(all_tokens) <max_seq_length:
                    ent_map.append([start_origin_ent, start_external_ent, len(all_tokens)])
        else:
            sub_tokens = tokenizer.tokenize(token)
            if not sub_tokens:
                sub_tokens = ["[UNK]"]
            all_tokens.extend(sub_tokens)
            tok_to_orig_index.extend([i] * len(sub_tokens))

            if ent_token_count % 3 != 2:
                pos_ids.append(pos_id)
                pos_id += 1

                if label_tags is None:
                    all_labels.extend(["" for _ in range(len(sub_tokens))])
                else:
                    try:
                        all_labels.extend([label_tags[i - ent_count] for _ in range(len(sub_tokens))])
                    except:
                        print()

            else:
                pos_ids.append(pos_id + ent_pos_id)
                ent_pos_id += 1
                ent_count += 1

                if label_tags is None:
                    all_labels.extend(["" for _ in range(len(sub_tokens))])
                else:
                    # 外部实体，不参与梯度回传
                    all_labels.extend(["" for _ in range(len(sub_tokens))])

    all_tokens = all_tokens[:max_seq_length - 1]
    all_labels = all_labels[:max_seq_length - 1]
    pos_ids = pos_ids[:max_seq_length - 1]
    all_tokens.append("[SEP]")
    all_labels.append("")
    tok_to_orig_index.append(-100)
    pos_ids.append(pos_ids[-1]+1)

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    label_ids = [label_map[label] if label else -100 for label in all_labels]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(-100)
        pos_ids.append(0)

    for i in range(len(ent_map)):
        s, m, e = ent_map[i]

        for etn_ent_id in range(m, e):
            for token_id in range(0, s):
                ent_visible_matrix[etn_ent_id][token_id] = 0
                ent_visible_matrix[token_id][etn_ent_id] = 0
            for token_id in range(e, max_seq_length):
                ent_visible_matrix[etn_ent_id][token_id] = 0
                ent_visible_matrix[token_id][etn_ent_id] = 0

    feature = KBERTLabelingFeatures(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_ids=label_ids,
                               all_tokens=all_tokens,
                               seq_length=max_seq_length,
                               tok_to_orig_index=tok_to_orig_index,
                               guid=example.guid,
                               visible_matrix=ent_visible_matrix,
                               position_ids=pos_ids
                               )

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
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         output_format="dict",
                         *args,
                         **kwargs)

        # assert ".easynlp/modelzoo/" in pretrained_model_name_or_path

        self.kbert_model_prefix = True if 'kbert' in pretrained_model_name_or_path else False

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        if self.kbert_model_prefix:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})
            kg_file = user_defined_parameters.get('kg_file', '')
            self.kg = KnowledgeGraph(spo_file=kg_file, predicate=True)

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

        if self.kbert_model_prefix:
            text_a = self.kg.add_knowledge_to_text(text_a)
            self.ent_id = self.tokenizer.convert_tokens_to_ids('[ENT]')

        text_b = None
        label = row[self.label_name] if self.label_name else None
        example = InputExample(text_a=text_a, text_b=text_b, label=label)
        if self.kbert_model_prefix:
            return kbert_labeling_convert_example_to_feature(example, self.tokenizer, self.max_seq_length, self.label_map)
        return bert_labeling_convert_example_to_feature(example, self.tokenizer,
                                                        self.max_seq_length, self.label_map)

    def batch_fn(self, features):
        if self.kbert_model_prefix:
            inputs = {
                "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
                "token_type_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
                "label_ids": torch.tensor([f.label_ids for f in features], dtype=torch.long),
                "tok_to_orig_index": [f.tok_to_orig_index for f in features],
                "visible_matrix": torch.tensor([f.visible_matrix for f in features], dtype=torch.long),
                "position_ids": torch.tensor([f.position_ids for f in features], dtype=torch.long)
            }
        else:
            inputs = {
                "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
                "token_type_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
                "label_ids": torch.tensor([f.label_ids for f in features], dtype=torch.long),
                "tok_to_orig_index": [f.tok_to_orig_index for f in features]
            }
        return inputs

class SequenceLabelingAutoDataset(GeneralDataset):
    """ Sequence Labeling Dataset base on GeneralDataset

    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file from 'load_dataset'
        max_seq_length: max sequence length of each input instance.

        The default setting of 'GeneralDataset' is implemented for SequenceClassification,
            so you need to choose the correct 'convert_single_row_to_example' and 'batch_fn' base on your application.

        In some special cases, you need to override the '__init__' function.

    """

    def convert_single_row_to_example(self, row):
        content_tokens = row[self.first_sequence]
        label_tags = row[self.label_name] if self.label_name else None
        all_tokens = ['[CLS]']
        all_labels = ['']
        tok_to_orig_index = [-100]
        for i, token in enumerate(content_tokens):
            sub_tokens = self.tokenizer.tokenize(token)
            if not sub_tokens:
                sub_tokens = ['[UNK]']
            all_tokens.extend(sub_tokens)
            tok_to_orig_index.extend([i] * len(sub_tokens))
            if label_tags is None:
                all_labels.extend(["" for _ in range(len(sub_tokens))])
            else:
                all_labels.extend([label_tags[i] for _ in range(len(sub_tokens))])
        all_tokens = all_tokens[:self.max_seq_length - 1]
        all_labels = all_labels[:self.max_seq_length - 1]
        all_tokens.append("[SEP]")
        all_labels.append("")
        tok_to_orig_index.append(-100)

        input_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        label_ids = [label if label != '' else -100 for label in all_labels]

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-100)

        feature = LabelingFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    label_ids=label_ids,
                                    all_tokens=all_tokens,
                                    seq_length=self.max_seq_length,
                                    tok_to_orig_index=tok_to_orig_index,
                                    guid=None)

        return feature

    def batch_fn(self, features):
        return SequenceLabelingDataset.batch_fn(self, features)


class KnowledgeGraph():
    """
        Construct KG structure for K-BERT
    """
    def __init__(self, spo_file, predicate=False, never_split_tag=None):
        self.predicate = predicate
        self.spo_file_paths = spo_file
        self.lookup_table = self._create_lookup_table()
        if never_split_tag:
            self.segment_vocab = list(self.lookup_table.keys()) + never_split_tag
        else:
            self.segment_vocab = list(self.lookup_table.keys())
        self.tokenizer = jieba
        for i in range(len(self.segment_vocab)):
            self.tokenizer.add_word(self.segment_vocab[i])
        self.special_tags = set(never_split_tag) if never_split_tag else None

    def _create_lookup_table(self):
        lookup_table = {}
        print("[KnowledgeGraph] Loading spo from {}".format(self.spo_file_paths))
        with open(self.spo_file_paths, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    subj, pred, obje = line.strip().split("\t")
                except:
                    print("[KnowledgeGraph] Bad spo:", line)
                if self.predicate:
                    value = pred + obje
                else:
                    value = obje
                if subj in lookup_table.keys():
                    lookup_table[subj].add(value)
                else:
                    lookup_table[subj] = set([value])
        return lookup_table

    def add_knowledge_to_text(self, sent, max_entities=2):
        sent = "".join(sent.split())
        split_sent = self.tokenizer.cut(sent)

        sent_tree = []
        know_sent = []

        for token in split_sent:

            entities = list(self.lookup_table.get(token, []))[:max_entities]
            entities = "".join(entities)
            sent_tree.append((token, entities))

        for i in range(len(sent_tree)):
            if len(sent_tree[i][1]) == 0:
                know_sent.extend([w for w in sent_tree[i][0]])
            elif len(sent_tree[i][1]) > 0:
                know_sent.append('[ENT]')
                know_sent.extend([w for w in sent_tree[i][0]])
                know_sent.append('[ENT]')
                know_sent.extend([w for w in sent_tree[i][1]])
                know_sent.append('[ENT]')

        row = " ".join(know_sent)


        return row