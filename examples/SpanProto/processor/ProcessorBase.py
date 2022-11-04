# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 5:10 pm.
# @Author  : JianingWang
# @File    : ProcessorBase.py
import json
import os.path

import numpy as np
from datasets import DatasetDict, Dataset, load_metric
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import DataCollatorWithPadding, EvalPrediction, DataCollatorForTokenClassification
from processor.dataset import DatasetK


class DataProcessor:
    def __init__(self, data_args, training_args, model_args):
        self.data_args = data_args
        self.training_args = training_args
        self.model_args = model_args
        if data_args.train_file:
            self.train_file = data_args.train_file
        if data_args.validation_file:
            self.dev_file = data_args.validation_file
        if data_args.test_file:
            self.test_file = data_args.test_file
        self.train_examples = None
        self.dev_examples = None
        self.test_examples = None
        self.tokenizer = None

    def get_examples(self, set_type):
        raise NotImplementedError()

    def get_data_collator(self):
        return NotImplementedError()

    def get_tokenized_datasets(self):
        return NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        return json.load(open(input_file, encoding='utf8'))

    @classmethod
    def _read_json2(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        return [json.loads(line) for line in open(input_file, encoding='utf8').readlines()]

    @classmethod
    def _read_text(cls, input_file):
        return open(input_file, encoding='utf8').readlines()

    @classmethod
    def _read_tsv(cls, input_file):
        return [i.strip().split('\t') for i in open(input_file, encoding='utf8').readlines()]

    @classmethod
    def list_2_json(cls, examples):
        # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        keys = list(examples[0].keys())
        json_data = {}
        for key in keys:
            json_data[key] = [e[key] for e in examples]
        return json_data

    def set_config(self, config):
        pass

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class CLSProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=False):
        super().__init__(data_args, training_args, model_args)
        self.sentence1_key = 'text_a'
        self.sentence2_key = None
        self.label_column = 'label'
        self.post_tokenizer = post_tokenizer
        self.keep_raw_data = keep_raw_data
        self.raw_datasets = None

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorWithPadding(self.tokenizer, padding='longest', pad_to_multiple_of=8 if pad_to_multiple_of_8 else None)

    def build_preprocess_function(self):
        # Tokenize the texts
        sentence1_key, sentence2_key = self.sentence1_key, self.sentence2_key
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length
        label_to_id = self.label_to_id
        label_column = self.label_column

        def func(examples):
            if sentence2_key:
                inputs = (examples[sentence1_key], examples[sentence2_key])
            else:
                inputs = (examples[sentence1_key],)
            result = tokenizer(*inputs,
                               padding=False,
                               max_length=max_seq_length,
                               truncation='longest_first',
                               add_special_tokens=True
                               )
            # result['token_type_ids'] = [[0]*len(i) for i in result['input_ids']]

            if label_to_id:
                # 分类
                result["label"] = [(label_to_id[l] if l else None) for l in examples[label_column]]
            else:
                # 回归
                result["label"] = [(l) for l in examples[label_column]]
            return result

        return func

    def get_tokenized_datasets(self):
        raw_datasets = DatasetDict()
        if self.training_args.do_train:
            train_examples = self.get_examples('train')
            raw_datasets['train'] = DatasetK.from_dict(self.list_2_json(train_examples)) # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        if self.training_args.do_eval:
            dev_examples = self.get_examples('dev')
            raw_datasets['validation'] = DatasetK.from_dict(self.list_2_json(dev_examples))
        if self.training_args.do_predict:
            test_examples = self.get_examples('test')
            raw_datasets['test'] = DatasetK.from_dict(self.list_2_json(test_examples))

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets
        # datasets的bug, 对于from_dict不会创建cache,需要指定cache_file_names
        # 指定了cache_file_names在_map_single中也需要cache_files不为空才能读取cache
        for key, value in raw_datasets.items():
            value.set_cache_files(['cache_local'])
        remove_columns = self.sentence1_key if not self.sentence2_key else [self.sentence1_key, self.sentence2_key]
        tokenize_func = self.build_preprocess_function()
        # 多gpu, 0计算完存cache，其他load cache
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir, 'datasets') if self.model_args.cache_dir else os.path.join(os.path.expanduser("~"), '.cache/huggingface/datasets/')
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name)

        os.makedirs(cache_dir, exist_ok=True)
        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                desc="Running tokenizer on dataset",
                cache_file_names={k: f'{cache_dir}/cache_{self.data_args.task_name}_{self.data_args.data_dir.split("/")[-1]}_{str(k)}.arrow' for k in raw_datasets},
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets

    def get_examples(self, set_type):
        raise NotImplementedError()

    def compute_metrics(self, p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = (preds == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        precision = precision_score(y_true=labels, y_pred=preds)
        return {
            'acc': round(acc, 4),
            'f1': round(f1, 4),
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'score': round(f1, 4)
        }







class FewShotProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=False):
        super().__init__(data_args, training_args, model_args)
        self.support_key = 'support_input_texts'
        self.query_key = 'query_input_texts'
        # self.input_column = 'input_texts'
        # self.label_column = 'label'
        self.post_tokenizer = post_tokenizer
        self.keep_raw_data = keep_raw_data
        self.raw_datasets = None
        self.mode = None
        self.num_class = None
        self.num_example = None

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorWithPadding(self.tokenizer, padding='longest', pad_to_multiple_of=8 if pad_to_multiple_of_8 else None)

    def build_preprocess_function(self):
        # Tokenize the texts
        support_key, query_key = self.support_key, self.query_key
        # input_column = self.input_column
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length
        # label_to_id = self.label_to_id
        # label_column = self.label_column

        '''
        examples = {
            'id': [...]
            'support_input_texts': [
                ['sent1', 'sent2', ...], # episode1
                ['sent1', 'sent2', ...], # episode2
                ...
            ]
            'support_labeled_spans': [[[x, x], xxx], ...],
            'support_labeled_types': [[..]...],
            'support_sentence_num': [...],
            'query_input_texts': ['xxx', 'xxx'],
            'query_labeled_spans': [[...]...],
            'query_labeled_types': [[..]...],
            'query_sentence_num': [...],
        }
        '''

        def func(examples):
            features = {
                'id': examples['id'],
                'support_input': list(),
                'query_input': list(),
            }
            support_inputs, query_inputs = examples[support_key], examples[query_key]
            for ei, support_input in enumerate(support_inputs):
                # 对每个episode，对support和query进行分词
                support_input = (support_input, )
                query_input = (query_inputs[ei], )
                support_result = tokenizer(
                    *support_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation='longest_first',
                    add_special_tokens=True
                )
                query_result = tokenizer(
                    *query_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation='longest_first',
                    add_special_tokens=True
                )
                features[support_input].append(support_result)
                features[query_input].append(query_result)
            features['support_labeled_spans'] = examples['support_labeled_spans']
            features['support_labeled_types'] = examples['support_labeled_types']
            features['support_sentence_num'] = examples['support_sentence_num']
            features['query_labeled_spans'] = examples['query_labeled_spans']
            features['query_labeled_types'] = examples['query_labeled_types']
            features['query_sentence_num'] = examples['query_sentence_num']

            return features

        return func

    def get_tokenized_datasets(self):
        raw_datasets = DatasetDict()
        if self.training_args.do_train:
            train_examples = self.get_examples('train')
            raw_datasets['train'] = DatasetK.from_dict(
                self.list_2_json(train_examples))  # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        if self.training_args.do_eval:
            dev_examples = self.get_examples('dev')
            raw_datasets['validation'] = DatasetK.from_dict(self.list_2_json(dev_examples))
        if self.training_args.do_predict:
            test_examples = self.get_examples('test')
            raw_datasets['test'] = DatasetK.from_dict(self.list_2_json(test_examples))

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets
        # datasets的bug, 对于from_dict不会创建cache,需要指定cache_file_names
        # 指定了cache_file_names在_map_single中也需要cache_files不为空才能读取cache
        for key, value in raw_datasets.items():
            value.set_cache_files(['cache_local'])
        remove_columns = [self.support_key, self.query_key]
        tokenize_func = self.build_preprocess_function()
        # 多gpu, 0计算完存cache，其他load cache
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir,
                                      'datasets') if self.model_args.cache_dir else os.path.join(
            os.path.expanduser("~"), '.cache/huggingface/datasets/')
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name)

        os.makedirs(cache_dir, exist_ok=True)
        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                desc="Running tokenizer on dataset",
                cache_file_names={
                    k: f'{cache_dir}/cache_{self.data_args.task_name}_{self.data_args.data_dir.split("/")[-1]}_{self.mode}_{self.num_class}_{self.num_example}_{str(k)}.arrow'
                    for k in raw_datasets},
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            print("datasets=", raw_datasets)
            return raw_datasets

    def get_examples(self, set_type):
        raise NotImplementedError()

    def compute_metrics(self, p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = (preds == labels).mean()
        f1 = f1_score(y_true=labels, y_pred=preds)
        recall = recall_score(y_true=labels, y_pred=preds)
        precision = precision_score(y_true=labels, y_pred=preds)
        return {
            'acc': round(acc, 4),
            'f1': round(f1, 4),
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'score': round(f1, 4)
        }

