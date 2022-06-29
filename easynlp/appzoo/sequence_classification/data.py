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

from ...distillation.distill_dataset import DistillatoryBaseDataset
from ...fewshot_learning.fewshot_dataset import FewshotBaseDataset
from ...modelzoo import AutoTokenizer
from ...utils import io
from ..dataset import BaseDataset


class ClassificationDataset(BaseDataset):
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
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)

        self.kbert_model_prefix = True if 'kbert' in pretrained_model_name_or_path else False

        # assert ".easynlp/modelzoo/" in pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        if self.kbert_model_prefix:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})
            kg_file = user_defined_parameters.get('kg_file', '')
            self.kg = KnowledgeGraph(spo_file=kg_file, predicate=True)

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

        if self.kbert_model_prefix:
            text_a = self.kg.add_knowledge_to_text(text_a)
            text_b = self.kg.add_knowledge_to_text(text_b) if self.second_sequence else None
            self.ent_id = self.tokenizer.convert_tokens_to_ids('[ENT]')

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

        if self.kbert_model_prefix:
            encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask'], encoding['position_ids'], encoding['visible_matrix'] = self.kbert_row_data_process(encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask'])


        return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}


    def kbert_row_data_process(self, input_ids, token_type_ids, attention_mask):
        """
            data process for K-BERT
        """

        ent_input_ids = []
        ent_token_type_ids = []
        ent_attention_mask = []
        ent_pos_ids = []
        ent_visible_matrix = [[1 for _ in range(self.max_seq_length)] for _ in range(self.max_seq_length)]

        pos_id = 0
        ent_pos_id = 0
        ent_map = [] # ent_map is the log of origin_ent_id and external_ent_id
        ent_token_count = 0
        start_origin_ent = None
        start_external_ent = None

        for i in range(len(input_ids)):
            input_id = input_ids[i]
            if input_id == self.ent_id:
                ent_token_count += 1
                ent_pos_id = 0
                if ent_token_count % 3 == 1:
                    start_origin_ent = len(ent_input_ids)
                elif ent_token_count % 3 == 2:
                    start_external_ent = len(ent_input_ids)
                else:
                    ent_map.append([start_origin_ent, start_external_ent, len(ent_input_ids)])
            else:
                ent_input_ids.append(input_id)
                ent_token_type_ids.append(token_type_ids[i])
                ent_attention_mask.append(attention_mask[i])
                if ent_token_count % 3 != 2:
                    ent_pos_ids.append(pos_id)
                    pos_id += 1
                else:
                    ent_pos_ids.append(pos_id+ent_pos_id)
                    ent_pos_id += 1

        if len(ent_input_ids) < self.max_seq_length:
            diff_length = self.max_seq_length-len(ent_input_ids)
            ent_input_ids.extend([0 for _ in range(diff_length)])
            ent_token_type_ids.extend(0 for _ in range(diff_length))
            ent_pos_ids.extend(0 for _ in range(diff_length))
            ent_attention_mask.extend(0 for _ in range(diff_length))

        for i in range(len(ent_map)):
            s, m, e = ent_map[i]

            for etn_ent_id in range(m, e):
                for token_id in range(0, s):
                    ent_visible_matrix[etn_ent_id][token_id] = 0
                    ent_visible_matrix[token_id][etn_ent_id] = 0
                for token_id in range(e, self.max_seq_length):
                    ent_visible_matrix[etn_ent_id][token_id] = 0
                    ent_visible_matrix[token_id][etn_ent_id] = 0

        return ent_input_ids, ent_token_type_ids, ent_attention_mask, ent_pos_ids, ent_visible_matrix


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
        split_sent = self.tokenizer.cut(sent)

        sent_tree = []
        know_sent = []

        for token in split_sent:

            entities = list(self.lookup_table.get(token, []))[:max_entities]
            entities = "".join(entities)
            sent_tree.append((token, entities))

        for i in range(len(sent_tree)):
            if len(sent_tree[i][1]) == 0:
                know_sent.append(sent_tree[i][0])
            elif len(sent_tree[i][1]) > 0:
                know_sent.append('[ENT]')
                know_sent.append(sent_tree[i][0])
                know_sent.append('[ENT]')
                know_sent.append(sent_tree[i][1])
                know_sent.append('[ENT]')

        row = "".join(know_sent)


        return row




class DistillatoryClassificationDataset(DistillatoryBaseDataset, ClassificationDataset):
    pass

class FewshotSequenceClassificationDataset(FewshotBaseDataset):
    pass
