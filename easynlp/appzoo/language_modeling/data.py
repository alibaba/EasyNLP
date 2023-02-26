# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and The HuggingFace Inc. team and  Facebook, Inc
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

from itertools import count
import random
import re
from tqdm import tqdm
from tracemalloc import start
from typing import Any, Dict, List, Union
import torch
import json
import pandas as pd
import numpy as np
import copy

from ..dataset import BaseDataset
from ...modelzoo import AutoTokenizer


class LanguageModelingDataset(BaseDataset):
    """ Whole word mask Language Model Dataset

    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file.
        max_seq_length: max sequence length of each input instance.
        mlm_mask_prop: the percentage of masked words
    """

    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 user_defined_parameters,
                 mlm_mask_prop=0.15,
                 **kwargs):
        super().__init__(data_file, **kwargs)
        # assert ".easynlp/modelzoo/" in pretrained_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(vocab)
        self.cls_ids = vocab["[CLS]"]
        self.pad_idx = vocab["[PAD]"]
        self.mask_idx = vocab["[MASK]"]
        self.sep_ids = vocab["[SEP]"]
        self.fp16 = False
        self.mlm_mask_prop = mlm_mask_prop
        self.max_seq_length = max_seq_length

        # DKPLM needs special tokens to recognize entity in input sentence
        self.dkplm_model_prefix = True if 'dkplm' in pretrained_model_name_or_path else False
        self.kangaroo_model_prefix = True if 'kangaroo' in pretrained_model_name_or_path else False
         # External knowledge
        self.external_konwledge_flag = kwargs.get('external_mask', False)
        self.Knowledge_G = kwargs.get('knowledge_graph', None)
        self.contrast_learning_flag = kwargs.get('contrast_learning_flag', False)
        self.negative_number = int(user_defined_parameters.get('negative_e_number', 0))
        self.negative_e_length = int(user_defined_parameters.get('negative_e_length', 0))
        if self.external_konwledge_flag:
             self.tokenizer.add_special_tokens({'additional_special_tokens': ['[dep]', '[sdp]']})
             
        if self.dkplm_model_prefix:
            entity_emb_file = user_defined_parameters.get('entity_emb_file', '')
            rel_emb_file = user_defined_parameters.get('rel_emb_file', '')
            if entity_emb_file == '' or rel_emb_file == '':
                raise ValueError('DKPLM needs knowledge embedding file...')
            entity_emb = []
            with open(entity_emb_file, 'r') as fin:
                for line in tqdm(fin, desc='loading entity embedding file...'):
                    vec = list(line.strip().split(','))
                    vec = [float(x) for x in vec]
                    entity_emb.append(vec)
            entity_emb = torch.FloatTensor(entity_emb)
            entity_emb = torch.nn.Embedding.from_pretrained(entity_emb)

            rel_emb = []
            with open(rel_emb_file, 'r') as fin:
                for line in tqdm(fin, desc='loading relation embedding file...'):
                    vec = list(line.strip().split(','))
                    vec = [float(x) for x in vec]
                    rel_emb.append(vec)
            rel_emb = torch.FloatTensor(rel_emb)
            rel_emb = torch.nn.Embedding.from_pretrained(rel_emb)
            self.entity_emb = entity_emb
            self.rel_emb = rel_emb
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})

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

            # create entity tree
            self.entity_tree, self.tokenid2entityid = self.kangaroo_create_entity_tree(entity_df)
            self.tokenidVec, self.positionidVec = self.kangaroo_get_contrastive_samples(CL_samples_file)
            self.conceptEmbVec = self.kangaroo_get_concept_emb(concept_emb_file)
    def get_positive_and_negative_examples(
        self,
        ner_data: str,
        negative_level: int = 3) -> Union[bool, Dict[str, List[str]]]:
        """get the positive examples and negative examples for the ner data

        Args:
            ner_data (str): the ner entity
            negative_level (int, optional): the depth of the relationship. Defaults to 3.

        Returns:
            Union[bool, Dict[str, List[str]]]: if the `ner_data` not in `knowledge`, return False, otherwise, return the positive and negative examples
        """

        knowledge: Dict[str, Dict[str, str]] = self.Knowledge_G
        common_used = set()
        def get_data(key: str, 
                    data: Dict[str, str], 
                    results: List[str], 
                    deep: int, 
                    insert_flag: bool = False):
            """get the negative examples recursively

            Args:
                key (str): the ner
                data (Dict[str, str]): the related data about `key`
                results (List[str]): a list used to save the negative examples
                deep (int): the recursive number
                insert_flag (bool, optional): whether insert data to `results`. Defaults to False.
            """
            nonlocal knowledge
            # Avoid data interference between different generations, such as: 汤恩伯：三民主义;国民党; 国民党:三民主义 二阶和一阶数据重复了
            common_used.add(key)
            if deep == 0:
                return
            else:
                for key_item in data:
                    if data[key_item] not in common_used and insert_flag == True:
                        results.append(data[key_item])
                    if data[key_item] in knowledge and data[key_item] not in common_used:
                        get_data(data[key_item], knowledge[data[key_item]], results, deep - 1, True)
        
        all_examples = {
            'ner': ner_data,
            'positive_examples': [],
            'negative_examples': []
        }
        
        if ner_data in knowledge:
            tp_data = knowledge[ner_data]
            negative_examples = []
            if '描述' in tp_data:
                positive_example = tp_data['描述']
            else:
                keys = list(tp_data.keys())
                choice = np.random.choice([_ for _ in range(len(keys))], 1)[0]
                positive_example = tp_data[keys[choice]]
            # # the description usually contains the ner entity, if not, concatenate the `ner_data` and the positive example
            if ner_data in positive_example:
                all_examples['positive_examples'].append(positive_example)
            else:
                all_examples['positive_examples'].append(ner_data + positive_example)
            
            get_data(ner_data, tp_data, negative_examples, negative_level)
            # concatenate the ner entity and each negative example
            negative_examples = list(map(lambda x: ner_data + x if ner_data not in x else x, negative_examples))
            all_examples['negative_examples'] = negative_examples
            return all_examples
            
        return False
    
    def positive_negative_examples_postprocess(self, 
                                       orginal_data: str, 
                                       ner_data: List[Dict[str, Any]], 
                                       positive_number:int = 1, 
                                       negative_number: int = 10) -> List[List[str]]:
        """get a specific number of positive and negative examples

        Args:
            orginal_data (str): the original data which is used to calculate the id index
            ner_data (List[Dict[str, Any]]): ner entity
            positive_number (int, optional): the positive examples. Defaults to 1.
            negative_number (int, optional): the negative examples. Defaults to 10.

        Returns:
            List[List[str]]: return the sampled positive examples and negative examples
        """
        sub_str = re.compile('\[CLS\]|\[sdp\]|\[dep\]')
        orginal_data = sub_str.sub('#', orginal_data)
        
        positive_examples = []
        negative_examples = []
        
        exist_ners = set()
        for ner_item in ner_data:
            if ner_item and ner_item['ner'] not in exist_ners:
                tp_ner = ner_item['ner']
                if tp_ner in orginal_data:
                    search_results = [tp_ner]
                    exist_ners.add(tp_ner)
                    copy_data = list(orginal_data)
                    for search_item in search_results:
                        ner_insert_data_p = []
                        ner_insert_data_n = []
                        start = ''.join(copy_data).index(search_item)
                        end = start + len(tp_ner)
                        ids = []
                        for _ in range(start, end):
                            if copy_data[_] != '#':
                                ids.append(_) 
                            copy_data[_] = '#'
                        for line_p in ner_item['positive_examples'][:positive_number]:
                            ner_insert_data_p.append([ids, line_p])
                        for line_n in ner_item['negative_examples'][:negative_number]:
                            ner_insert_data_n.append([ids, line_n])
                    
                        positive_examples.append(ner_insert_data_p)
                        negative_examples.append(ner_insert_data_n)

        if len(positive_examples) * len(negative_examples) > 0:
            return [positive_examples, negative_examples]
        else:
            return None
        ...
    
    def ckbert_row_data_process(self, row):
        """get processed examples

        Args:
            row (_type_): the original data

        Returns:
            _type_: _description_
        """
        # positive number is set to 1
        positive_number = 1
        negative_number = self.negative_number
        example_number = self.negative_e_length
        text_line = eval(row)
        token_ids = [self.cls_ids]
        positive_negative_example_token_ids = []
        if self.contrast_learning_flag:
            positive_negative_example_token_ids = [[[self.pad_idx] * example_number for _ in range(positive_number + negative_number)] for _ in range(self.max_seq_length)]
            ner_data_dicts = []
            deepth = 3
            text_copy = ''.join(text_line[0]).replace('[sdp]', '')
            text_copy = text_copy.replace('[dep]', '')
            ner_exist = set()
            try:
                for ner_str in text_line[2]:
                    if ner_str in text_copy and ner_str not in ner_exist:
                        ner_exist.add(ner_str)
                        ner_data_dicts.append(self.get_positive_and_negative_examples(ner_str, deepth))
            except:
                ...
        
            if len(ner_data_dicts) > 0:
                positive_negative_examples = self.positive_negative_examples_postprocess(''.join(text_line[0]), ner_data_dicts, positive_number, negative_number)
                
                if positive_negative_examples:
                    try:
                        for example_item in positive_negative_examples:
                            start = 0
                            for example_data in example_item:
                                counter = start
                                for example in example_data:
                                    tp_length = len(example[1])
                                    tp_ids = []
                                    for index in range(min(tp_length, example_number)):
                                        token = self.tokenizer.tokenize(example[1][index])
                                        id_ = self.tokenizer.convert_tokens_to_ids(token)
                                        tp_ids.extend(id_)
                                    gap = example_number - len(tp_ids)
                                    for example_id in example[0]:
                                        if len(tp_ids) > example_number:
                                            positive_negative_example_token_ids[example_id][counter] = tp_ids[:example_number]
                                        else:
                                            positive_negative_example_token_ids[example_id][counter] = tp_ids + [0] * gap
                                    counter += 1
                            start += 1
                    except:
                        ...
            
        for sentence in text_line[0][1:-1]:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            id_ = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            token_ids.extend(id_)
        token_ids.append(self.sep_ids)
        return token_ids, text_line[1], [], positive_negative_example_token_ids
        
    def convert_single_row_to_example(self, row):
        if self.dkplm_model_prefix:
            text_line = eval(row)
            # entity_id and ent_pos are in the same order
            text, relation_id, replaced_entity_id = text_line['text'], \
                                                    text_line['relation_id'], \
                                                    text_line['replced_entity_id']
            token_ids = [self.cls_ids]
            sentence_tokens, ent_pos = self.dkplm_row_data_process(text)
            token_ids.extend(self.tokenizer.convert_tokens_to_ids(sentence_tokens))
            ent_pos = [[item[0]+1, item[1]+1] for item in ent_pos]
        elif self.kangaroo_model_prefix:
            return self.kangaroo_row_data_process(row)
        elif self.external_konwledge_flag:
            return self.ckbert_row_data_process(row)
        else:
            text = json.loads(row.strip())['text']
            token_ids = [self.cls_ids]
            for sentence in text:
                sentence_tokens = self.tokenizer.tokenize(sentence)
                token_ids.extend(self.tokenizer.convert_tokens_to_ids(sentence_tokens))
        token_ids = token_ids[:self.max_seq_length - 1]
        token_ids.append(self.sep_ids)
        ref_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        mask_labels, mask_span_indices = self._whole_word_mask(ref_tokens)
        if self.dkplm_model_prefix:
            return token_ids, mask_labels, ent_pos, relation_id, replaced_entity_id, mask_span_indices
        else:
            return token_ids, mask_labels, mask_span_indices

    def batch_fn(self, batch):
        if self.kangaroo_model_prefix:
            input_ids = [t[0] for t in batch]
            attention_mask = [t[1] for t in batch]
            label_ids = [t[2] for t in batch]
            entities_position = [t[3] for t in batch]
            ent_mask = [t[4] for t in batch]
            sample_token_id = [t[5] for t in batch]
            sample_position_id = [t[6] for t in batch]
            sample_mask = [t[7] for t in batch]
            concept_emb = [t[8] for t in batch]

            # input_ids = [t[0] for t in batch]
            # attention_mask = [t[1] for t in batch]
            # label_ids = [t[2] for t in batch]
            # entities_position = [t[3] for t in batch]
            # ent_mask = [t[4] for t in batch]
            # sample_token_id = [t[5] for t in batch]
            # sample_position_id = [t[6] for t in batch]
            # sample_mask = [t[7] for t in batch]
            # concept_emb = [t[8] for t in batch]

            return {
                'input_ids': torch.LongTensor(input_ids),
                'attention_mask': torch.LongTensor(attention_mask),
                'label_ids': torch.LongTensor(label_ids),
                'entities_position': torch.LongTensor(entities_position),
                'ent_mask': torch.LongTensor(ent_mask),
                'sample_token_id': torch.LongTensor(sample_token_id),
                'sample_position_id': torch.LongTensor(sample_position_id),
                'sample_mask': torch.LongTensor(sample_mask),
                'concept_emb': torch.LongTensor(concept_emb)
            }
        token_ids = [t[0] for t in batch]
        mask_labels = [t[1] for t in batch]
        if self.contrast_learning_flag:
            positive_negative_ids = [t[3] for t in batch]
                
            for _index, item in enumerate(batch):
                t_length = len(item[0])
                if t_length > self.max_seq_length:
                    gap = t_length - self.max_seq_length
                    token_ids[_index] = batch[_index][0][:t_length - gap - 1] + [batch[_index][0][-1]]
                    mask_labels[_index] = batch[_index][1][:t_length - gap - 1] + [batch[_index][1][-1]]
        lengths = [len(t) for t in token_ids]
        # Max for paddings
        max_seq_len_ = max(lengths)
        # max_seq_len_ = self.max_seq_length
        assert max_seq_len_ <= self.max_seq_length
        if self.dkplm_model_prefix:
            max_seq_len_ = self.max_seq_length
            ent_pos = [t[2] for t in batch]
            relation_id = [t[3] for t in batch]
            replaced_entity_id = [t[4] for t in batch]
            
            insert_know_position_mask, insert_know_emb, \
            insert_relation_emb, \
            insert_know_labels = self.align_dkplm_input(max_seq_len_, \
                                token_ids, ent_pos, \
                                relation_id, \
                                replaced_entity_id)

        # Pad token ids
        padded_token_ids = [t + [self.pad_idx] * (self.max_seq_length - len(t)) for t in token_ids]
        padded_mask_labels = [t + [self.pad_idx] * (self.max_seq_length - len(t)) for t in mask_labels]
        assert len(padded_token_ids) == len(token_ids)
        assert all(len(t) == self.max_seq_length for t in padded_token_ids)
        assert all(len(t) == self.max_seq_length for t in padded_mask_labels)

        token_ids = torch.LongTensor(padded_token_ids)
        mask_labels = torch.LongTensor(padded_mask_labels)
        lengths = torch.tensor(lengths)  # (bs)
        if self.dkplm_model_prefix:
            insert_know_position_mask = torch.LongTensor(insert_know_position_mask)
            insert_know_emb = insert_know_emb
            insert_relation_emb = insert_relation_emb
            insert_know_labels = torch.LongTensor(insert_know_labels)
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        attn_mask = attn_mask.long()

        input_ids, label_ids = self.mask_tokens(token_ids, mask_labels)

        if self.dkplm_model_prefix:
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "label_ids": label_ids,
                "insert_know_position_mask": insert_know_position_mask,
                "insert_know_emb": insert_know_emb,
                "insert_relation_emb": insert_relation_emb,
                "insert_know_labels": insert_know_labels,
                "mask_span_indices": [t[5] for t in batch],
                "return_dict": False
            }
        elif self.contrast_learning_flag:
            
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "label_ids": label_ids,
                "mask_span_indices": [t[2] for t in batch],
                "positive_negative_examples":torch.tensor(positive_negative_ids, device=lengths.device),
                # "positive_negative_ner_mask":positive_negative_ner_masks
            } 
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "label_ids": label_ids,
                "mask_span_indices": [t[2] for t in batch]
            }

    def _whole_word_mask(self, input_tokens, max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions,
                             max(1, int(round(len(input_tokens) * self.mlm_mask_prop))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

        mask_span_indices = [t for t in cand_indexes if t[0] in covered_indexes]
        return mask_labels, mask_span_indices

    def mask_tokens(self, inputs, mask_labels):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        Set 'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        # special_tokens_mask = [
        #     self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        # ]
        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        padding_mask = labels.eq(self.pad_idx)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def dkplm_row_data_process(self, sentence):
        sentence_tokens = self.tokenizer.tokenize(sentence)
        start_flag = False
        ent_pos = []
        start_pos = -1
        end_pos = -1
        for index, tokens in enumerate(sentence_tokens):
            if tokens == '[ENT]':
                if start_flag:
                    start_flag = False
                    end_pos = index
                    ent_pos.append([start_pos, end_pos])
                else:
                    start_flag = True
                    start_pos = index
        if start_pos == -1 or end_pos == -1:
            raise ValueError("This pretraining sentence in DKPLM hasn't entities!")
        else:
            new_ent_pos = []
            for ent_pair_index, item in enumerate(ent_pos):
                if ent_pair_index == 0:
                    new_ent_pos.append([item[0], item[1]-1])
                else:
                    # ent = sentence_tokens[item[0], item[1]]
                    new_ent_pos.append([item[0]-2*ent_pair_index, item[1]-1-ent_pair_index*2])
            sentence_tokens = list(filter(lambda x:x!='[ENT]', sentence_tokens))
            return sentence_tokens, new_ent_pos

    def align_dkplm_input(self, max_seq_len, token_ids, ent_pos, relation_id, replaced_entity_id):
        insert_know_position_mask=[]
        insert_know_labels = []
        insert_know_labels_zeros = []
        insert_rel_labels_zeros = []
        zip_index = -1
        for token_id, ent_pos_id in zip(token_ids, ent_pos):
            zip_index += 1
            temp_position_mask = [0]*len(token_id)
            temp_know_labels = [-100] * len(token_id)
            temp_know_labels_zeros = [0] * len(token_id)
            temp_rel_labels_zeros = [0] * len(token_id)

            for index, start_end_item in enumerate(ent_pos_id):
                start_entity, end_entity = start_end_item
                temp_position_mask = temp_position_mask[:start_entity] + \
                                        [1] * (end_entity-start_entity) + \
                                        temp_position_mask[end_entity:]
                temp_know_labels[start_entity:end_entity] = token_id[start_entity:end_entity]
                temp_know_labels_zeros[start_entity:end_entity] = [replaced_entity_id[zip_index][index] for _ in range(start_entity, end_entity)]
                temp_rel_labels_zeros[start_entity:end_entity] = [relation_id[zip_index][index] for _ in range(start_entity, end_entity)]

            insert_know_position_mask.append(temp_position_mask)
            insert_know_labels.append(temp_know_labels)
            insert_know_labels_zeros.append(temp_know_labels_zeros)
            insert_rel_labels_zeros.append(temp_rel_labels_zeros)

        padded_insert_know_labels = [t + [-100] * (max_seq_len - len(t)) for t in insert_know_labels]
        padded_insert_know_labels_zeros = [t + [0] * (max_seq_len - len(t)) for t in insert_know_labels_zeros]
        padded_insert_know_position_mask = [t + [0] * (max_seq_len - len(t)) for t in insert_know_position_mask]
        padded_insert_rel_labels_zeros = [t + [0] * (max_seq_len - len(t)) for t in insert_rel_labels_zeros]
        
        # batch * max_seq * emb_size
        padded_entity_emb = self.entity_emb(torch.LongTensor(padded_insert_know_labels_zeros))

        # batch * max_seq * emb_size
        insert_rel_emb = torch.ones((len(token_ids), max_seq_len, self.entity_emb.weight.size()[1]))
        replaced_insert_rel_emb = torch.zeros((len(token_ids), max_seq_len, self.entity_emb.weight.size()[1]))
        padded_rel_emb = self.rel_emb(torch.LongTensor(padded_insert_rel_labels_zeros))
        # decoder loss module: r * h, so default value of padded_rel_emb is 1 not 0
        padded_rel_emb = torch.where(padded_rel_emb!=0., padded_rel_emb, insert_rel_emb)
        # replaced entity emb = h + r, so default value is 0 not 1
        replaced_padded_rel_emb = torch.where(padded_rel_emb!=0., padded_rel_emb, replaced_insert_rel_emb)
        
        # replaced entity = entity + relation (TransE)
        padded_replaced_entity_emb = padded_entity_emb + replaced_padded_rel_emb

        return padded_insert_know_position_mask, padded_replaced_entity_emb, padded_rel_emb, padded_insert_know_labels

    def kangaroo_row_data_process(self, text, entity_num=3, entity_gap=5):

        tokens = [t for t in text]
        # tokens = self.tokenizer.tokenize([t for t in text])
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(token_ids) > self.max_seq_length - 2:
            token_ids = token_ids[:(self.max_seq_length - 2)]

        # entity position
        entity_pos = []
        i = 0
        while i < len(token_ids):
            search_result = self.entity_tree.search(token_ids, i)
            if len(search_result) == 0:
                i = i + 1
                continue
            j = search_result[-1]
            entity = token_ids[i:j]
            entity_pos.append((i, j))
            i = j + 1

        # entity id list
        entities = [-100 for _ in range(len(token_ids))]
        # entity id list, transform id to 1,2,3...n
        entities_position = [0 for _ in range(len(token_ids))]
        entity_index = 0
        entity_pos_true = []

        entity_id_list = []
        # 保存实体首、尾位置的index。[ent1_head_index, ent1_tail_index, ent2_head_index,...]
        entity_head_tail = []

        for pos in entity_pos:
            close_flag = False
            h_index = pos[0]
            t_index = pos[1]
            # 保证entity之间间隔>=entity_gap, debug check entity_pos是否从小到大顺序
            for i in range(1, entity_gap+1):
                if h_index - i < 0:
                    continue
                if entities[h_index - i] != -100:
                    close_flag = True
            if close_flag:
                continue
            entity_id = self.tokenid2entityid[str(token_ids[h_index:t_index])]
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

        masked_tokens_id, masked_lm_labels = self.kangaroo_create_mask(token_ids, entity_pos_true)

        input_mask = list((np.array(token_ids) != -1) * 1)

        # input_ids = [self.cls_ids] + token_ids + [self.sep_ids]
        masked_tokens_id = [self.cls_ids] + masked_tokens_id + [self.sep_ids]
        entities = [-100] + entities + [-100]
        # 检查entities_position第0位要不要非0
        entities_position = [0] + entities_position + [0]
        masked_lm_labels = [-100] + masked_lm_labels + [-100]
        input_mask = [1] + input_mask + [1]

        if len(masked_tokens_id) < self.max_seq_length:
            rest = self.max_seq_length - len(masked_tokens_id)
            masked_tokens_id.extend([0] * rest)
            entities.extend([-100] * rest)
            entities_position.extend([0] * rest)
            masked_lm_labels.extend([-100] * rest)
            input_mask.extend([0] * rest)

        # 补全padding
        assert len(masked_tokens_id) == len(entities_position)
        assert len(entities_position) == len(masked_lm_labels)

        # masked_tokens_id = torch.LongTensor(masked_tokens_id)
        # input_mask = torch.LongTensor(input_mask)
        # masked_lm_labels = torch.LongTensor(masked_lm_labels)
        entities_position = torch.LongTensor(entities_position)

        ent_mask = torch.LongTensor((entities_position != 0) * 1)
        entity_id_index = torch.LongTensor(entity_id_list) + 1
        sample_token_id = self.tokenidVec[entity_id_index]
        sample_position_id = self.positionidVec[entity_id_index]
        sample_mask = torch.LongTensor((np.array(sample_token_id) != 0) * 1)
        concept_emb = self.conceptEmbVec[entity_id_index]  # [batch_size, entity_num, concept_size]

        return masked_tokens_id, input_mask, masked_lm_labels, entities_position.tolist(), ent_mask.tolist(), sample_token_id.tolist(), sample_position_id.tolist(), sample_mask.tolist(), concept_emb.tolist()

    def kangaroo_create_mask(self, tokens_id, entity_pos_true, entity_gap=5):

        entity_prop = 0.1
        masked_lm_labels = [-100 for _ in range(len(tokens_id))]
        masked_tokens_id = copy.deepcopy(tokens_id)

        input_len = len(tokens_id)
        entities_length = np.sum([j - i for (i, j) in entity_pos_true])

        while entities_length / input_len > entity_prop:
            del entity_pos_true[random.randint(0, len(entity_pos_true) - 1)]
            entities_length = np.sum([j - i for (i, j) in entity_pos_true])

        entity_probability = entities_length / input_len
        # 考虑entity 前后距离较近的不进行mlm
        mlm_token_probability = (self.mlm_mask_prop - entity_probability) * input_len / (input_len - 7 * len(entity_pos_true))

        # entity masking
        token_mlm_flag = [1 for _ in range(len(tokens_id))]
        for po in entity_pos_true:
            masked_lm_labels[po[0]:po[1]] = tokens_id[po[0]:po[1]]
            masked_tokens_id[po[0]:po[1]] = [self.mask_idx] * (po[1] - po[0])
            if po[0] - entity_gap < 0:
                s_index = 0
            else:
                s_index = po[0] - entity_gap
            if po[1] + entity_gap > len(tokens_id):
                e_index = len(tokens_id)
            else:
                e_index = po[1] + entity_gap
            token_mlm_flag[s_index: e_index] = [0] * (e_index - s_index)

        # token masking
        for ind in range(len(token_mlm_flag)):
            if token_mlm_flag[ind] == 0:
                continue

            if random.random() > mlm_token_probability:
                continue

            if random.random() < 0.8:
                masked_tokens_id[ind] = self.mask_idx
            else:
                if random.random() < 0.5:
                    masked_tokens_id[ind] = tokens_id[ind]
                else:
                    masked_tokens_id[ind] = random.randint(0, self.vocab_size - 1)

            masked_lm_labels[ind] = tokens_id[ind]

        return masked_tokens_id, masked_lm_labels

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