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

import os
from typing import Dict, Union, List
from easynlp.appzoo.api import get_application_model, get_application_dataset, get_application_evaluator
from easynlp.appzoo.dataset import BaseDataset
from datasets import load_dataset as hf_load_dataset
from easynlp.modelzoo import AutoConfig, AutoTokenizer
from easynlp.utils.logger import logger


class DatasetPreprocessor:
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            max_seq_length: int = 128,
            is_training: bool = True,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.padding = "max_length"
        self.max_seq_length = max_seq_length
        self.is_training = is_training

        self.column_name_dict = self.get_column_name()
        self.remove_column_list = [
            column for key, value in self.column_name_dict.items() for column in value
        ]
        self.label2id = {label_name: label_id for label_id, label_name in enumerate(self.get_labels())}
        self.id2label = {label_id: label_name for label_id, label_name in enumerate(self.get_labels())}


    def convert_examples_to_features(self, examples):
        '''
        unified features convertor
        :param examples: datasets examples
        :return: features
        '''
        column_name_dict = self.column_name_dict
        input_text1, input_text2 = list(), None
        if len(column_name_dict['text']) == 1:
            text_column_name = column_name_dict['text'][0]
            input_text1 = [
                value for value in examples[text_column_name] if len(value) > 0 and not None
            ]
        else:
            text_column_name1, text_column_name2 = column_name_dict['text'][0], column_name_dict['text'][1]
            input_text1 = [
                value for value in examples[text_column_name1] if len(value) > 0 and not None
            ]
            input_text2 = [
                value for value in examples[text_column_name2] if len(value) > 0 and not None
            ]

        features = self.tokenizer(
            input_text1,
            text_pair=input_text2,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )
        # print('features.keys()=', features.keys())

        if 'label' in column_name_dict.keys() or self.is_training:
            label_column_name = column_name_dict['label'][0]
            # print('label_column_name=', label_column_name)
            try:
                label_ids = [
                    self.label2id[value] for value in examples[label_column_name] if len(value) > 0 and not None
                ]
                features['label_ids'] = label_ids
            except:
                print('This could be test example.')

        return features

    def get_labels(self) -> List:
        pass

    def get_column_name(self) -> Dict:
        pass

    def get_app_name(self) -> str:
        pass


class AfqmcProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        return ["0", "1"]

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['sentence1', 'sentence2'], 'label': ['label'], 'other': []}
        return {'text': ['sentence1', 'sentence2'], 'other': []}

    def get_app_name(self) -> str:
        return "text_match"


class TnewsProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['sentence'], 'label': ['label'], 'other': ['label_desc', 'keywords']}
        return {'text': ['sentence'], 'other': ['label_desc', 'keywords']}

    def get_app_name(self) -> str:
        return "text_classify"


class IflytekProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        labels = []
        for i in range(119):
            labels.append(str(i))
        return labels

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['sentence'], 'label': ['label'], 'other': ['label_des']}
        return {'text': ['sentence'], 'other': ['label_des']}


    def get_app_name(self) -> str:
        return "text_classify"


class OcnliProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        return ["contradiction", "entailment", "neutral"]

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['sentence1', 'sentence2'], 'label': ['label'], 'other': []}
        return {'text': ['sentence1', 'sentence2'], 'other': []}

    def get_app_name(self) -> str:
        return "text_match"

    def convert_examples_to_features(self, examples):
        '''
        unified features convertor
        :param examples: datasets examples
        :return: features
        '''
        column_name_dict = self.column_name_dict
        label_ids = None
        if 'label' in column_name_dict.keys() or self.is_training:
            label_column_name = column_name_dict['label'][0]
            # print('label_column_name=', label_column_name)
            try:
                label_ids = [
                    self.label2id[value] if value != "-" else "-" for value in examples[label_column_name]
                ]
            except:
                print('This could be test example.')

        text_column_name1, text_column_name2 = column_name_dict['text'][0], column_name_dict['text'][1]
        input_text1 = [
            value for ei, value in enumerate(examples[text_column_name1])
            if len(value) > 0 and (label_ids is None or (label_ids is not None and label_ids[ei] != "-"))
        ]
        input_text2 = [
            value for ei, value in enumerate(examples[text_column_name2])
            if len(value) > 0 and (label_ids is None or (label_ids is not None and label_ids[ei] != "-"))
        ]

        features = self.tokenizer(
            input_text1,
            text_pair=input_text2,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )

        if label_ids is not None:
            features['label_ids'] = [value for value in label_ids if value != "-"]

        return features


class CmnliProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        return ["contradiction", "entailment", "neutral"]

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['sentence1', 'sentence2'], 'label': ['label'], 'other': []}
        return {'text': ['sentence1', 'sentence2'], 'other': []}

    def get_app_name(self) -> str:
        return "text_match"

    def convert_examples_to_features(self, examples):
        '''
        unified features convertor
        :param examples: datasets examples
        :return: features
        '''
        column_name_dict = self.column_name_dict
        label_ids = None
        if 'label' in column_name_dict.keys() or self.is_training:
            label_column_name = column_name_dict['label'][0]
            # print('label_column_name=', label_column_name)
            try:
                label_ids = [
                    self.label2id[value] if value != "-" else "-" for value in examples[label_column_name]
                ]
            except:
                print('This could be test example.')

        text_column_name1, text_column_name2 = column_name_dict['text'][0], column_name_dict['text'][1]
        input_text1 = [
            value for ei, value in enumerate(examples[text_column_name1])
            if len(value) > 0 and (label_ids is None or (label_ids is not None and label_ids[ei] != "-"))
        ]
        input_text2 = [
            value for ei, value in enumerate(examples[text_column_name2])
            if len(value) > 0 and (label_ids is None or (label_ids is not None and label_ids[ei] != "-"))
        ]

        features = self.tokenizer(
            input_text1,
            text_pair=input_text2,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )
        if label_ids is not None:
            features['label_ids'] = [value for value in label_ids if value != "-"]

        return features


class CslProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        return ["0", "1"]

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['abst'], 'candidate': ['keyword'], 'label': ['label'], 'other': ['id']}
        return {'text': ['abst'], 'candidate': ['keyword'], 'other': []}

    def get_app_name(self) -> str:
        return "text_match"

    def convert_examples_to_features(self, examples):
        column_name_dict = self.column_name_dict
        input_text1, input_text2 = list(), None
        input_text1 = [
            " ".join(value) for value in examples[column_name_dict['candidate'][0]] if len(value) > 0 and not None
        ]
        input_text2 = [
            value for value in examples[column_name_dict['text'][0]] if len(value) > 0 and not None
        ]

        features = self.tokenizer(
            input_text1,
            text_pair=input_text2,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )

        if 'label' in column_name_dict.keys() or self.is_training:
            label_column_name = column_name_dict['label'][0]
            try:
                label_ids = [
                    self.label2id[value] for value in examples[label_column_name] if len(value) > 0 and not None
                ]
                features['label_ids'] = label_ids
            except:
                logger.info('This could be test example.')

        return features


class WscProcessor(DatasetPreprocessor):

    def get_labels(self) -> List:
        return ["true", "false"]

    def get_column_name(self) -> Dict:
        if self.is_training:
            return {'text': ['text'], 'span': ['target'], 'label': ['label'], 'other': ['idx']}
        return {'text': ['text'], 'span': ['target'], 'other': []}

    def get_app_name(self) -> str:
        return "text_classify"

    def convert_examples_to_features(self, examples):
        '''
        unified features convertor
        :param examples: datasets examples
        :return: features
        '''
        column_name_dict = self.column_name_dict
        input_text1, input_text2 = list(), None
        text_column_name = column_name_dict['text'][0]
        span_column_name = column_name_dict['span'][0]
        input_text1 = list()

        for ei, text_a in enumerate(examples[text_column_name]):
            if len(text_a) == 0:
                continue
            text_a_list = list(text_a)
            target = examples[span_column_name][ei]
            query = target['span1_text']
            query_idx = target['span1_index']
            pronoun = target['span2_text']
            pronoun_idx = target['span2_index']
            assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
            if pronoun_idx > query_idx:
                text_a_list.insert(query_idx, "_")
                text_a_list.insert(query_idx + len(query) + 1, "_")
                text_a_list.insert(pronoun_idx + 2, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_a_list.insert(pronoun_idx, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_a_list.insert(query_idx + 2, "_")
                text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
            text_a = "".join(text_a_list)
            input_text1.append(text_a)

        features = self.tokenizer(
            input_text1,
            text_pair=input_text2,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )
        # print('features.keys()=', features.keys()

        if 'label' in column_name_dict.keys() or self.is_training:
            label_column_name = column_name_dict['label'][0]
            # print('label_column_name=', label_column_name)
            try:
                label_ids = [
                    self.label2id[value] for value in examples[label_column_name] if len(value) > 0 and not None
                ]
                features['label_ids'] = label_ids
            except:
                print('This could be test example.')

        return features


tasks2processor = {
    'afqmc': AfqmcProcessor,
    'tnews': TnewsProcessor,
    'iflytek': IflytekProcessor,
    'ocnli': OcnliProcessor,
    'cmnli': CmnliProcessor,
    'csl': CslProcessor,
    'wsc': WscProcessor,
}