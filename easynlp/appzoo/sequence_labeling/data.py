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
import pandas as pd
import jieba
import random

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
                 position_ids=None,
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


class KangarooLabelingFeatures(object):
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
                 entities_position=None,
                 ent_mask=None,
                 sample_token_id=None,
                 sample_position_id=None,
                 sample_mask=None,
                 concept_emb=None
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.all_tokens = all_tokens
        self.seq_length = seq_length
        self.label_ids = label_ids
        self.tok_to_orig_index = tok_to_orig_index
        self.guid = guid
        self.entities_position = entities_position
        self.ent_mask = ent_mask
        self.sample_token_id = sample_token_id
        self.sample_position_id = sample_position_id
        self.sample_mask = sample_mask
        self.concept_emb = concept_emb


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


def kangaroo_labeling_convert_example_to_feature(example, tokenizer, max_seq_length, entity_tree, tokenid2entityid, tokenidVec, positionidVec, conceptEmbVec, entity_num=3, entity_gap=5, label_map=None):
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


    token_ids = input_ids
    entity_pos = []
    i = 0
    while i < len(token_ids):
        search_result = entity_tree.search(token_ids, i)
        if len(search_result) == 0:
            i = i + 1
            continue
        j = search_result[-1]
        entity = token_ids[i:j]
        entity_pos.append((i, j))
        i = j + 1

    # entity id list
    entities = [-100 for _ in range(len(token_ids))]
    # entity id list, trasform id to 1,2,3...n
    entities_position = [0 for _ in range(len(token_ids))]
    entity_index = 0
    entity_pos_true = []

    entity_id_list = []
    entity_head_tail = []

    for pos in entity_pos:
        close_flag = False
        h_index = pos[0]
        t_index = pos[1]
        for i in range(1, entity_gap+1):
            if h_index - i < 0:
                continue
            if entities[h_index - i] != -100:
                close_flag = True
        if close_flag:
            continue
        entity_id = tokenid2entityid[str(token_ids[h_index:t_index])]
        entity_index += 1
        entity_pos_true.append(pos)

        entity_id_list.append(entity_id)
        entity_head_tail.extend([h_index, t_index - 1])

        for ent_index in range(h_index, t_index):
            entities[ent_index] = entity_id
            entities_position[ent_index] = entity_index

        if entity_index == entity_num:
            break

    if entity_index < entity_num:
        for j in range(entity_num - entity_index):
            entity_id_list.append(-1)
            entity_head_tail.extend([-1, -1])

    input_mask = list((np.array(token_ids) != -1) * 1)


    entities_position = torch.LongTensor(entities_position)

    ent_mask = torch.LongTensor((entities_position != 0) * 1)
    entity_id_index = torch.LongTensor(entity_id_list) + 1
    sample_token_id = tokenidVec[entity_id_index]
    sample_position_id = positionidVec[entity_id_index]
    sample_mask = torch.LongTensor((np.array(sample_token_id) != 0) * 1)
    concept_emb = conceptEmbVec[entity_id_index]  # [batch_size, entity_num, concept_size]

    feature = KangarooLabelingFeatures(input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_ids=label_ids,
                                       all_tokens=all_tokens,
                                       seq_length=max_seq_length,
                                       tok_to_orig_index=tok_to_orig_index,
                                       guid=example.guid,
                                       entities_position=entities_position.tolist(),
                                       ent_mask=ent_mask.tolist(),
                                       sample_token_id=sample_token_id.tolist(),
                                       sample_position_id=sample_position_id.tolist(),
                                       sample_mask=sample_mask.tolist(),
                                       concept_emb=concept_emb.tolist())

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

        self.kangaroo_model_prefix = True if 'kangaroo' in pretrained_model_name_or_path else False

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        if self.kbert_model_prefix:
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})
            kg_file = user_defined_parameters.get('kg_file', '')
            self.kg = KnowledgeGraph(spo_file=kg_file, predicate=True)

        self.max_seq_length = max_seq_length

        if self.kangaroo_model_prefix:
            entity_file = user_defined_parameters.get('entity_file', '')
            rel_file = user_defined_parameters.get('rel_file', '')
            CL_samples_file = user_defined_parameters.get('samples_file', '')
            concept_emb_file = user_defined_parameters.get('concept_emb_file', '')
            if entity_file == '' or rel_file == '':
                raise ValueError('Kangaroo needs knowledge embedding file...')

            rel_df = pd.read_csv(rel_file)
            # entity_df = pd.read_csv(entity_file)[:500]
            entity_df = pd.read_csv(entity_file)
            self.entity_tree, self.tokenid2entityid = self.kangaroo_create_entity_tree(entity_df)
            self.tokenidVec, self.positionidVec = self.kangaroo_get_contrastive_samples(CL_samples_file)
            self.conceptEmbVec = self.kangaroo_get_concept_emb(concept_emb_file)

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
        if self.kangaroo_model_prefix:
            return kangaroo_labeling_convert_example_to_feature(example, self.tokenizer, self.max_seq_length, self.entity_tree,
                                                                self.tokenid2entityid, self.tokenidVec, self.positionidVec, self.conceptEmbVec, label_map=self.label_map)
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
        elif self.kangaroo_model_prefix:
            inputs = {
                "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
                "attention_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
                "token_type_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
                "label_ids": torch.tensor([f.label_ids for f in features], dtype=torch.long),
                "tok_to_orig_index": [f.tok_to_orig_index for f in features],
                "entities_position": torch.tensor([f.entities_position for f in features], dtype=torch.long),
                "ent_mask": torch.tensor([f.ent_mask for f in features], dtype=torch.long),
                "sample_token_id": torch.tensor([f.sample_token_id for f in features], dtype=torch.long),
                "sample_position_id": torch.tensor([f.sample_position_id for f in features], dtype=torch.long),
                "sample_mask": torch.tensor([f.sample_mask for f in features], dtype=torch.long),
                "concept_emb": torch.tensor([f.concept_emb for f in features]),
                "pretrain_model": [False],

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

    def kangaroo_create_entity_tree(self, entity_df):
        full_name_to_id = {}
        for i in range(len(entity_df)):
            full_name = entity_df.iloc[i]['main_name']
            name_list = entity_df.iloc[i]['name_list'].split('|')
            if pd.isna(full_name):
                name_list = entity_df.iloc[i]['name_list'].split('|')
            id = int(entity_df.iloc[i]['index'])
            for name in name_list:
                full_name_to_id[name] = id

        entities = list(full_name_to_id.keys())
        entities_tokens_id = []
        tokenid2entityid = {}
        for entity in entities:
            entity_token_id = self.tokenizer.convert_tokens_to_ids([k for k in entity])
            entities_tokens_id.append(entity_token_id)
            tokenid2entityid[str(entity_token_id)] = full_name_to_id[entity]
        entity_tree = KangarooTrieTree()
        for word in entities_tokens_id:
            entity_tree.add_word(word)
        return entity_tree, tokenid2entityid

    def kangaroo_get_contrastive_samples(self, samples_file, max_level=4):
        samples = np.load(samples_file, allow_pickle=True).item()
        max_index = np.max(list(samples.keys()))
        token_id_vec = [[[0 for _ in range(self.max_seq_length)] for _ in range(max_level)] for _ in
                        range(max_index + 2)]
        pos_id_vec = [[[0 for _ in range(self.max_seq_length)] for _ in range(max_level)] for _ in range(max_index + 2)]
        # for ind in random.sample(samples.keys(), 500):
        for ind in samples.keys():
            try:
                token_id_list = []
                pos_id_list = []
                for le in range(1, max_level + 1):
                    level = "level_%d" % le
                    if len(samples[ind][level]) == 0:
                        level = "level_2"
                    tokens = samples[ind][level][0]['tokens']
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    pos_ids = samples[ind][level][0]['position_id']
                    # assert len(token_ids) == len(pos_ids)

                    if len(token_ids) < self.max_seq_length:
                        token_ids.extend([0] * (self.max_seq_length - len(token_ids)))
                        pos_ids.extend([0] * (self.max_seq_length - len(pos_ids)))

                    token_id_list.append(token_ids)
                    pos_id_list.append(pos_ids)

                token_id_vec[ind + 1] = token_id_list
                pos_id_vec[ind + 1] = pos_id_list
            except:
                continue

        return torch.LongTensor(token_id_vec), torch.LongTensor(pos_id_vec)

    def kangaroo_get_concept_emb(self, emb_file, dim=100):
        entity2emb = np.load(emb_file, allow_pickle=True).item()
        max_index = np.max(list(entity2emb.keys()))
        concept_emb_vec = [[0 for _ in range(dim)] for _ in range(int(max_index) + 2)]
        for ind in entity2emb.keys():
            concept_emb_vec[int(ind) + 1] = entity2emb[ind]
        return torch.FloatTensor(concept_emb_vec)


class KangarooTrieTree:
    """
        Construct entity prefix structure for KANGAROO
    """
    def __init__(self):
        self.node = [""]
        self.edge = [{}]
        self.flag = [False]

    def add_node(self, node):
        self.node.append(node)
        self.edge.append({})
        self.flag.append(False)
        return len(self.node) - 1

    def add_word(self, word):
        u = 0
        for i in word:
            if i not in self.edge[u]:
                self.edge[u][i] = self.add_node(i)
            u = self.edge[u][i]
        self.flag[u] = True

    def show(self):
        for i in range(len(self.node)):
            print(i)
            print(self.node[i])
            print(self.edge[i])
            print(self.flag[i])
            print()

    def search(self, sentence, start_position):
        i = start_position
        u = 0
        result = []
        while i < len(sentence) and sentence[i] in self.edge[u]:
            u = self.edge[u][sentence[i]]
            i += 1
            if self.flag[u]:
                result.append(i)
        return result

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