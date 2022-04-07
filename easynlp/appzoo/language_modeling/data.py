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
from re import L
from tqdm import tqdm
from tracemalloc import start
import torch
import json

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
        token_ids = [t[0] for t in batch]
        mask_labels = [t[1] for t in batch]
        lengths = [len(t[0]) for t in batch]
        # Max for paddings
        max_seq_len_ = max(lengths)
        assert max_seq_len_ <= self.max_seq_length
        if self.dkplm_model_prefix:
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
        padded_token_ids = [t + [self.pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
        padded_mask_labels = [t + [self.pad_idx] * (max_seq_len_ - len(t)) for t in mask_labels]
        assert len(padded_token_ids) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in padded_token_ids)
        assert all(len(t) == max_seq_len_ for t in padded_mask_labels)

        token_ids = torch.LongTensor(padded_token_ids)
        mask_labels = torch.LongTensor(padded_mask_labels)
        lengths = torch.tensor(lengths)  # (bs)
        if self.dkplm_model_prefix:
            insert_know_position_mask = torch.LongTensor(insert_know_position_mask)
            insert_know_emb = insert_know_emb
            insert_relation_emb = insert_relation_emb
            insert_know_labels = torch.LongTensor(insert_know_labels)
        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long,
                                 device=lengths.device) < lengths[:, None]
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
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
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
        # replcaed entity emb = h + r, so default valur is 0 not 1
        replaced_padded_rel_emb = torch.where(padded_rel_emb!=0., padded_rel_emb, replaced_insert_rel_emb)
        
        # replcaed entity = entity + relation (TransE)
        padded_replaced_entity_emb = padded_entity_emb + replaced_padded_rel_emb

        return padded_insert_know_position_mask, padded_replaced_entity_emb, padded_rel_emb, padded_insert_know_labels