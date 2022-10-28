# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 10:19 pm.
# @Author  : JianingWang
# @File    : data_process.py
import json
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from transformers import PreTrainedTokenizerBase
from processor.ProcessorBase import CLSProcessor, FewShotProcessor
from collections import defaultdict, Counter
from torch import distributed as dist

from datasets import DatasetDict
from processor.dataset import DatasetK

def generate_global_pointer_labels(data_set: dict, max_length):
    # data_set: support/query set
    labeled_spans, labeled_types, offset_mappings = data_set['labeled_spans'], data_set['labeled_types'], data_set['offset_mapping']
    new_labeled_spans, new_labeled_types = list(), list()
    labels = torch.zeros(
        len(labeled_spans), 1, max_length, max_length
    )


    for ei in range(len(labeled_spans)): 
        labeled_span = labeled_spans[ei]
        labeled_type = labeled_types[ei]
        offset = offset_mappings[ei]
        new_labeled_span, new_labeled_type = list(), list()
        # starts, ends = feature['start'], feature['end']
        # print('starts=', starts)
        # print('ends=', ends)
        position_map = {}
        for i, (m, n) in enumerate(offset):
            if i != 0 and m == 0 and n == 0:
                continue
            for k in range(m, n + 1):
                position_map[k] = i
        if len(labeled_span) == 0:
            labels[ei, 0, 0, 0] = 1
            new_labeled_span.append([])
            new_labeled_types.append([])
        for ej, span in enumerate(labeled_span):
            start, end = span
            end -= 1
            # if start == 0:
            #     # assert end == -1
            #     labels[ei, 0, 0, 0] = 1
            #     new_labeled_span.append([0, 0])

            if start in position_map and end in position_map:
                labels[ei, 0, position_map[start], position_map[end]] = 1
                new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_type.append(labeled_type[ej])
        new_labeled_spans.append(new_labeled_span)
        new_labeled_types.append(new_labeled_type)
    return labels, new_labeled_spans, new_labeled_types



@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 64
    num_class: Optional[int] = None
    num_example: Optional[int] = 5
    mode: Optional[str] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None
    path: Optional[bool] = None

    def __call__(self, features):
        '''
            input:
            features = [
                {
                    'id': xx
                    'support_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxxx
                    },
                    'query_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxx
                    },
                    'support_labeled_spans': [[[x, x], ..], ..],
                    'support_labeled_types': [[xx, ..], ..],
                    'support_sentence_num': xx,
                    'query_labeled_spans': [[[x, x], ..], ..],
                    'query_labeled_types': [[xx, ..], ..],
                    'query_sentence_num': xx,
                    'stage': xx,
                }
            ]

            return
            features = {
                'support': {
                    'input_ids': [],
                    'attention_mask': [[xxx], ...],
                    'token_type_ids': [[xxx], ...],
                    'labeled_spans':,
                    'labeled_types':,
                    'sentence_num': [xx, ...],
                    'labels': []
                },
                'query': {
                    'input_ids': [],
                    'attention_mask': [[xxx], ...],
                    'token_type_ids': [[xxx], ...],
                    'labeled_spans':,
                    'labeled_types':,
                    'sentence_num':,
                    'labels': []
                },
                'num_class': x,
            }
        '''
        id_batch = list()
        support_batch = {
            'input_ids': list(), 'attention_mask': list(), 'token_type_ids': list(), 'offset_mapping': list(),
            'labeled_spans': list(), 'labeled_types': list(), 'sentence_num': list(), 'labels': list()
        }
        query_batch = {
            'input_ids': list(), 'attention_mask': list(), 'token_type_ids': list(), 'offset_mapping': list(),
            'labeled_spans': list(), 'labeled_types': list(), 'sentence_num': list(), 'labels': list()
        }

        # all_support_sentence_num, all_query_sentence_num = 0, 0
        stage = features[0]['stage']

        # if stage == "dev":
        #     print('0 collator stage=', stage)
        for feature_id, feature in enumerate(features):
            # print(feature)
            id_batch.append(feature['id'])
            if 'num_class' in feature.keys():
                self.num_class = feature['num_class']
                # print('self.num_class', self.num_class)
            support_input, query_input = feature['support_input'], feature['query_input']
            support_input = self.tokenizer.pad(
                support_input,
                padding='max_length', 
                max_length=self.max_length,
            )

            query_input = self.tokenizer.pad(
                query_input,
                padding='max_length',
                max_length=self.max_length,
            )

            support_batch['input_ids'].extend(support_input['input_ids'])
            support_batch['attention_mask'].extend(support_input['attention_mask'])
            support_batch['token_type_ids'].extend(support_input['token_type_ids'])
            support_batch['offset_mapping'].extend(support_input['offset_mapping'])

            query_batch['input_ids'].extend(query_input['input_ids'])
            query_batch['attention_mask'].extend(query_input['attention_mask'])
            query_batch['token_type_ids'].extend(query_input['token_type_ids'])
            query_batch['offset_mapping'].extend(query_input['offset_mapping'])

            support_batch['labeled_spans'].extend(feature['support_labeled_spans'])
            support_batch['labeled_types'].extend(feature['support_labeled_types'])
            support_batch['sentence_num'].append(feature['support_sentence_num'])

            query_batch['labeled_spans'].extend(feature['query_labeled_spans'])
            query_batch['labeled_types'].extend(feature['query_labeled_types'])
            query_batch['sentence_num'].append(feature['query_sentence_num'])

        support_labels, support_new_labeled_spans, support_new_labeled_types = generate_global_pointer_labels(support_batch, self.max_length)
        query_labels, query_new_labeled_spans, query_new_labeled_types = generate_global_pointer_labels(query_batch, self.max_length)
        support_batch.pop('offset_mapping')
        query_batch.pop('offset_mapping')

        support_batch['labeled_spans'] = support_new_labeled_spans
        support_batch['labeled_types'] = support_new_labeled_types
        query_batch['labeled_spans'] = query_new_labeled_spans
        query_batch['labeled_types'] = query_new_labeled_types

        support_batch['labels'] = support_labels
        if support_batch['labels'].max() > 0:
            support_batch['short_labels'] = torch.ones(len(support_batch['labeled_spans']))
        else:
            support_batch['short_labels'] = torch.zeros(len(support_batch['labeled_spans']))

        query_batch['labels'] = query_labels
        if query_batch['labels'].max() > 0:
            query_batch['short_labels'] = torch.ones(len(query_batch['labeled_spans']))
        else:
            query_batch['short_labels'] = torch.zeros(len(query_batch['labeled_spans']))

        # convert to torch
        # id_batch = torch.Tensor(id_batch).long()
        support_batch['input_ids'] = torch.Tensor(support_batch['input_ids']).long()
        support_batch['attention_mask'] = torch.Tensor(support_batch['attention_mask']).long()
        support_batch['token_type_ids'] = torch.Tensor(support_batch['token_type_ids']).long()
        query_batch['input_ids'] = torch.Tensor(query_batch['input_ids']).long()
        query_batch['attention_mask'] = torch.Tensor(query_batch['attention_mask']).long()
        query_batch['token_type_ids'] = torch.Tensor(query_batch['token_type_ids']).long()

        # print('====')
        # if stage == "dev":
        #     print('1 collator stage=', stage)
        return {
            'episode_ids': id_batch,
            'support': support_batch,
            'query': query_batch,
            'num_class': self.num_class,
            'num_example': self.num_example,
            'mode': self.mode,
            'stage': stage,
            'short_labels': torch.zeros(len(features)),
            'path': self.path
        }


class FewNERDProcessor(FewShotProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")} # user_defined parameter
        N, Q, K, mode = param["N"], param["Q"], param["K"], param['mode'] # N: num class, Q: query entity num, K: support entity num
        self.train_file = os.path.join(data_args.data_dir, "train_{}_{}.jsonl".format(N, K))
        self.dev_file = os.path.join(data_args.data_dir, "dev_{}_{}.jsonl".format(N, K))
        self.test_file = os.path.join(data_args.data_dir, "test_{}_{}.jsonl".format(N, K))

        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.mode = mode
        self.num_class = int(N)
        self.num_example = int(K)

        self.output_dir = "./outputs/{}-{}-{}".format(self.mode, self.num_class, self.num_example)


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollator(
            self.tokenizer,
            num_class=self.num_class,
            num_example=self.num_example,
            mode=self.mode,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length,
        )

    def __load_data_from_file__(self, filepath):
        with open(filepath)as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
        return lines

    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self._create_examples(self.__load_data_from_file__(self.train_file), set_type)
            self.train_examples = examples
        elif set_type == 'dev':
            examples = self._create_examples(self.__load_data_from_file__(self.dev_file), set_type)
            self.dev_examples = examples
        elif set_type == 'test':
            examples = self._create_examples(self.__load_data_from_file__(self.test_file), set_type)
            self.test_file = examples
        else:
            examples = None
        return examples

    def get_sentence_with_span(self, data, label2id):
        word_list = data["word"]
        label_list = data["label"]
        input_texts = list()
        labeled_spans = list()
        labeled_types = list()
        for words, labels in zip(word_list, label_list):
            start, end = -1, -1
            current_label = ""
            text = ""
            spans = list()
            span_types = list()
            for ei, word in enumerate(words):
                label = labels[ei]
                if label == "O":
                    text += word + " "
                    if start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                else:
                    if label != current_label and start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                    if start == -1:
                        start = len(text)
                    text += word + " "
                    end = len(text)
                    current_label = label
            if start != -1:
                spans.append([start, end])
                span_types.append(label2id[current_label])
                # start, end = -1, -1
                # current_label = ""
            input_texts.append(text.strip())
            labeled_spans.append(spans)
            labeled_types.append(span_types)
        return input_texts, labeled_spans, labeled_types



    def _create_examples(self, lines, set_type):
        examples = []
        # is_train = 0 if set_type == 'test' else 1
        for id_, line in enumerate(lines):
            target_classes = line['types']
            label2id = {v: ei for ei, v in enumerate(target_classes)}
            support = line['support']
            query = line['query']
            support_input_texts, support_labeled_spans, support_labeled_types = self.get_sentence_with_span(support, label2id)
            query_input_texts, query_labeled_spans, query_labeled_types = self.get_sentence_with_span(query, label2id)

            examples.append(
                {
                    'id': id_,
                    'support_input_texts': support_input_texts,
                    'support_labeled_spans': support_labeled_spans,
                    'support_labeled_types': support_labeled_types,
                    'support_sentence_num': len(support_input_texts),
                    'query_input_texts': query_input_texts,
                    'query_labeled_spans': query_labeled_spans,
                    'query_labeled_types': query_labeled_types,
                    'query_sentence_num': len(query_input_texts),
                    'stage': set_type,
                }
            )

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True
        return config

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
        
        return
        
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
                support_input = (support_input, )
                query_input = (query_inputs[ei], )
                support_result = tokenizer(
                    *support_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation='longest_first',
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                query_result = tokenizer(
                    *query_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation='longest_first',
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                features['support_input'].append(support_result)
                features['query_input'].append(query_result)
            features['support_labeled_spans'] = examples['support_labeled_spans']
            features['support_labeled_types'] = examples['support_labeled_types']
            features['support_sentence_num'] = examples['support_sentence_num']
            features['query_labeled_spans'] = examples['query_labeled_spans']
            features['query_labeled_types'] = examples['query_labeled_types']
            features['query_sentence_num'] = examples['query_sentence_num']

            return features

        return func

    def fush_multi_answer(self, has_answer, new_answer):

        # has: {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
        # new {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
        # print('has_answer=', has_answer)
        for ans, value in new_answer.items():
            if ans not in has_answer.keys():
                has_answer[ans] = value
            else:
                has_answer[ans]['prob'] += value['prob']
                has_answer[ans]['pos'].extend(value['pos'])
        return has_answer



    def get_predict_result(self, logits, examples, stage='dev'):
        '''
            query_spans: list = None # e.g. [[[[1, 3], [6, 9]], ...], ...]
            proto_logits: list = None # e.g. [[[0, 3], ...], ...]
            topk_probs: torch.FloatTensor = None
            topk_indices: torch.IntTensor = None
            examples:
                {
                    'id': xx
                    'support_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxxx
                    },
                    'query_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxx
                    },
                    'support_labeled_spans': [[[x, x], ..], ..],
                    'support_labeled_types': [[xx, ..], ..],
                    'support_sentence_num': xx,
                    'query_labeled_spans': [[[x, x], ..], ..],
                    'query_labeled_types': [[xx, ..], ..],
                    'query_sentence_num': xx,
                    'stage': xx,
                }
        '''
        # query_spans, proto_logits, _, __ = logits
        word_size = dist.get_world_size()
        results = dict()
        for i in range(word_size):
            path = os.path.join(self.output_dir, "predict", "{}_predictions_{}.npy".format(stage, i))
            # path = "./outputs2/predict/predictions_{}.npy".format(i)
            assert os.path.exists(path), "unknown path: {}".format(path)
            if os.path.exists(path):
                res = np.load(path, allow_pickle=True)[()]
                for episode_i, value in res.items():
                    results[episode_i] = value

        predictions = dict()

        for example in examples:
            # episode ground truth
            query_labeled_spans = example['query_labeled_spans']
            query_labeled_types = example['query_labeled_types']
            query_offset_mapping = example['query_input']['offset_mapping']
            id_ = example['id']
            new_labeled_spans = list()
            for ei in range(len(query_labeled_spans)):
                labeled_span = query_labeled_spans[ei]
                offset = query_offset_mapping[ei]
                new_labeled_span = list()
                # starts, ends = feature['start'], feature['end']
                # print('starts=', starts)
                # print('ends=', ends)
                position_map = {}
                for i, (m, n) in enumerate(offset):
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i

                for span in labeled_span:
                    start, end = span
                    end -= 1
                    if start in position_map and end in position_map:
                        new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_spans.append(new_labeled_span)

            pred_spans, pred_spans_ = results[id_]["spans"], list()
            pred_types, pred_types_ = results[id_]["types"], list()

            predictions[id_] = {
                'labeled_spans': new_labeled_spans,
                'labeled_types': query_labeled_types,
                'predicted_spans': pred_spans,
                'predicted_types': pred_types
            }

        return predictions

    def compute_metrics(self, eval_predictions, stage='dev'):
        '''
        eval_predictions: huggingface
        eval_predictions[0]: logits
        eval_predictions[1]: labels
        # print("raw_datasets=", raw_datasets["validation"])
            Dataset({
                features: ['id', 'support_labeled_spans', 'support_labeled_types', 'support_sentence_num', 'query_labeled_spans', 'query_labeled_types', 'query_sentence_num', 'stage', 'support_input', 'query_input'],
                num_rows: 1000
            })
        '''
        all_metrics = {
            "span_precision": 0.,
            "span_recall": 0.,
            "eval_span_f1": 0,
            "class_precision": 0.,
            "class_recall": 0.,
            "eval_class_f1": 0,
        }

        examples = self.raw_datasets['validation'] if stage == "dev" else self.raw_datasets['test']
        # golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions = self.get_predict_result(eval_predictions[0], examples, stage)

        # === copy from Few-NERD metric ===

        pred_span_cnt = 0  # pred entity cnt
        label_span_cnt = 0  # true label entity cnt
        correct_span_cnt = 0  # correct predicted entity cnt

        pred_class_cnt = 0  # pred entity cnt
        label_class_cnt = 0  # true label entity cnt
        correct_class_cnt = 0  # correct predicted entity cnt

        for episode_id, predicts in predictions.items():
            query_labeled_spans = predicts['labeled_spans']
            query_labeled_types = predicts['labeled_types']
            pred_span = predicts['predicted_spans']
            pred_type = predicts['predicted_types']
            for label_span, label_type, pred_span, pred_type in zip(
                    query_labeled_spans, query_labeled_types, pred_span, pred_type
            ):
                label_span_dict = {0: list()}
                pred_span_dict = {0: list()}
                label_class_dict = dict()
                pred_class_dict = dict()
                for span, type in zip(label_span, label_type):
                    label_span_dict[0].append((span[0], span[1]))
                    if type not in label_class_dict.keys():
                        label_class_dict[type] = list()
                    label_class_dict[type].append((span[0], span[1]))
                for span, type in zip(pred_span, pred_type):
                    pred_span_dict[0].append((span[0], span[1]))
                    if type == self.num_class or span == [0, 0]:
                        continue
                    if type not in pred_class_dict.keys():
                        pred_class_dict[type] = list()
                    pred_class_dict[type].append((span[0], span[1]))

                tmp_pred_span_cnt, tmp_label_span_cnt, correct_span = self.metrics_by_entity(
                    label_span_dict, pred_span_dict
                )

                tmp_pred_class_cnt, tmp_label_class_cnt, correct_class = self.metrics_by_entity(
                    label_class_dict, pred_class_dict
                )
                pred_span_cnt += tmp_pred_span_cnt
                label_span_cnt += tmp_label_span_cnt
                correct_span_cnt += correct_span

                pred_class_cnt += tmp_pred_class_cnt
                label_class_cnt += tmp_label_class_cnt
                correct_class_cnt += correct_class

        span_precision = correct_span_cnt / pred_span_cnt
        span_recall = correct_span_cnt / label_span_cnt
        try:
            span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall)
        except:
            span_f1 = 0.

        class_precision = correct_class_cnt / pred_class_cnt
        class_recall = correct_class_cnt / label_class_cnt
        try:
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)
        except:
            class_f1 = 0.
        all_metrics['span_precision'], all_metrics['span_recall'], all_metrics['eval_span_f1'] = \
            span_precision, span_recall, span_f1
        all_metrics['class_precision'], all_metrics['class_recall'], all_metrics['eval_class_f1'] = \
            class_precision, class_recall, class_f1
        if stage == "dev":
            print("[development dataset] all_metrics=", all_metrics)
        else:
            # print("[testing dataset] all_metrics=", all_metrics)
            print("**** Testing Result ****")
            for key, value in all_metrics.items():
                print("{}:".format(key), value)
        return all_metrics

    def metrics_by_entity(self, label_class_span, pred_class_span):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        # pred_class_span # {label:[(start_pos, end_pos), ...]}
        # label_class_span # {label:[(start_pos, end_pos), ...]}
        def get_cnt(label_class_span):
            '''
            return the count of entities
            '''
            cnt = 0
            for label in label_class_span:
                cnt += len(label_class_span[label])
            return cnt

        def get_intersect_by_entity(pred_class_span, label_class_span):
            '''
            return the count of correct entity
            '''
            cnt = 0
            for label in label_class_span:
                cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
            return cnt

        pred_cnt = get_cnt(pred_class_span)
        label_cnt = get_cnt(label_class_span)
        correct_cnt = get_intersect_by_entity(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def save_result(self, logits, label_ids):
        self.compute_metrics((logits, ), stage='test')



class CrossNERProcessor(FewShotProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, post_tokenizer=post_tokenizer,
                         keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in
                 (data_args.user_defined).split(" ")}  # user_defined parameter
        N, K, ID, mode = param["N"], param["K"], param["ID"], param["mode"]  # N: num class, Q: query entity num, K: support entity num
        # mode = "xval_ner" if K == 1 else "x_val_ner_shot_5"
        # notes: in crossner, N denotes the num class of target domain
        self.train_file = os.path.join(
            data_args.data_dir, mode,
            ("ner_train_{}.json".format(ID)) if K == '1' else ("ner-train-{}-shot-5.json".format(ID))
        )
        self.dev_file = os.path.join(
            data_args.data_dir, mode,
            ("ner_valid_{}.json".format(ID)) if K == '1' else ("ner-valid-{}-shot-5.json".format(ID)))
        self.test_file = os.path.join(
            data_args.data_dir, mode,
            ("ner_test_{}.json".format(ID)) if K == '1' else ("ner-test-{}-shot-5.json".format(ID)))

        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.mode = mode
        self.num_class = None
        self.num_example = int(K)
        self.ID = ID

        self.output_dir = "./outputs/{}-{}".format(self.mode, ID)

    def get_num_class(self, data_labels):
        if data_labels == "News":
            return 4
        if data_labels == "Wiki":
            return 11
        if data_labels == "SocialMedia":
            return 6
        if data_labels == "OntoNotes":
            return 18


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollator(
            self.tokenizer,
            num_class=self.num_class,
            num_example=self.num_example,
            mode=self.mode,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length,
            path=self.output_dir
        )

    def __load_data_from_file__(self, filepath):
        with open(filepath) as f:
            raw_data = json.load(f)
        return raw_data


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
        for key, value in raw_datasets.items():
            value.set_cache_files(['cache_local'])
        remove_columns = [self.support_key, self.query_key]
        tokenize_func = self.build_preprocess_function()
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
                    k: f'{cache_dir}/cache_{self.data_args.task_name}_{self.data_args.data_dir.split("/")[-1]}_{self.mode}_{self.ID}_{self.num_example}_{str(k)}.arrow'
                    for k in raw_datasets},
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            print("datasets=", raw_datasets)
            return raw_datasets

    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self._create_examples(self.__load_data_from_file__(self.train_file), set_type)
            self.train_examples = examples
        elif set_type == 'dev':
            examples = self._create_examples(self.__load_data_from_file__(self.dev_file), set_type)
            self.dev_examples = examples
        elif set_type == 'test':
            examples = self._create_examples(self.__load_data_from_file__(self.test_file), set_type)
            self.test_file = examples
        else:
            examples = None
        return examples

    def get_sentence_with_span(self, data, label2id):
        word_list = data["seq_ins"]
        label_list = data["seq_outs"]
        input_texts = list()
        labeled_spans = list()
        labeled_types = list()
        for words, labels in zip(word_list, label_list):
            start, end = -1, -1
            current_label = ""
            text = ""
            spans = list()
            span_types = list()
            for ei, word in enumerate(words):
                label = labels[ei]
                label = label.replace("B-", "").replace("I-", "")
                if label == "O":
                    text += word + " "
                    if start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                else:
                    if label != current_label and start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                    if start == -1:
                        start = len(text)
                    text += word + " "
                    end = len(text)
                    current_label = label
            if start != -1:
                spans.append([start, end])
                span_types.append(label2id[current_label])
                # start, end = -1, -1
                # current_label = ""
            input_texts.append(text.strip())
            labeled_spans.append(spans)
            labeled_types.append(span_types)
        return input_texts, labeled_spans, labeled_types

    def _create_examples(self, raw_data: dict, set_type):

        def get_label2id(suport_labels, query_labels):
            label2id = dict()
            for sent in suport_labels + query_labels:
                for label in sent:
                    if label == "O":
                        continue
                    label = label.replace("B-", "").replace("I-", "")
                    if label not in label2id.keys():
                        label2id[label] = len(label2id)
            return label2id
        examples = []
        # is_train = 0 if set_type == 'test' else 1
        for domain_name, domain_data in raw_data.items():

            for id_, line in enumerate(domain_data):
                # label2id = {v: ei for ei, v in enumerate(target_classes)}
                support = line['support']
                query = line['batch']
                label2id = get_label2id(support['seq_outs'], query['seq_outs'])
                support_input_texts, support_labeled_spans, support_labeled_types = self.get_sentence_with_span(support,
                                                                                                                label2id)
                query_input_texts, query_labeled_spans, query_labeled_types = self.get_sentence_with_span(query, label2id)

                num_class = self.get_num_class(domain_name)
                assert num_class == len(label2id)

                examples.append(
                    {
                        'id': id_,
                        'support_input_texts': support_input_texts,
                        'support_labeled_spans': support_labeled_spans,
                        'support_labeled_types': support_labeled_types,
                        'support_sentence_num': len(support_input_texts),
                        'query_input_texts': query_input_texts,
                        'query_labeled_spans': query_labeled_spans,
                        'query_labeled_types': query_labeled_types,
                        'query_sentence_num': len(query_input_texts),
                        'num_class': num_class,
                        'stage': set_type,
                    }
                )

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True
        return config

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

        return

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
                support_input = (support_input,)
                query_input = (query_inputs[ei],)
                support_result = tokenizer(
                    *support_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation='longest_first',
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                query_result = tokenizer(
                    *query_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation='longest_first',
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                features['support_input'].append(support_result)
                features['query_input'].append(query_result)
            features['support_labeled_spans'] = examples['support_labeled_spans']
            features['support_labeled_types'] = examples['support_labeled_types']
            features['support_sentence_num'] = examples['support_sentence_num']
            features['query_labeled_spans'] = examples['query_labeled_spans']
            features['query_labeled_types'] = examples['query_labeled_types']
            features['query_sentence_num'] = examples['query_sentence_num']

            return features

        return func

    def fush_multi_answer(self, has_answer, new_answer):
        # has: {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
        # new {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
        # print('has_answer=', has_answer)
        for ans, value in new_answer.items():
            if ans not in has_answer.keys():
                has_answer[ans] = value
            else:
                has_answer[ans]['prob'] += value['prob']
                has_answer[ans]['pos'].extend(value['pos'])
        return has_answer

    def get_predict_result(self, logits, examples, stage='dev'):
        '''
        query_spans: list = None # e.g. [[[[1, 3], [6, 9]], ...], ...]
        proto_logits: list = None # e.g. [[[0, 3], ...], ...]
        topk_probs: torch.FloatTensor = None
        topk_indices: torch.IntTensor = None
        examples:
            {
                    'id': xx
                    'support_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxxx
                    },
                    'query_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxx
                    },
                    'support_labeled_spans': [[[x, x], ..], ..],
                    'support_labeled_types': [[xx, ..], ..],
                    'support_sentence_num': xx,
                    'query_labeled_spans': [[[x, x], ..], ..],
                    'query_labeled_types': [[xx, ..], ..],
                    'query_sentence_num': xx,
                    'num_class': xx,
                    'stage': xx,
                }
        '''
        # query_spans, proto_logits, _, __ = logits
        word_size = dist.get_world_size()
        num_class = examples[0]['num_class']
        results = dict()
        for i in range(word_size):
            path = os.path.join(
                self.output_dir, "predict", "{}_predictions_{}.npy".format(stage, i))
            # path = "./outputs2/predict/predictions_{}.npy".format(i)
            assert os.path.exists(path), "unknown path: {}".format(path)
            if os.path.exists(path):
                res = np.load(path, allow_pickle=True)[()]
                for episode_i, value in res.items():
                    results[episode_i] = value

        predictions = dict()

        for example in examples:
            query_labeled_spans = example['query_labeled_spans']
            query_labeled_types = example['query_labeled_types']
            query_offset_mapping = example['query_input']['offset_mapping']
            num_class = example['num_class']
            id_ = example['id']
            new_labeled_spans = list()
            for ei in range(len(query_labeled_spans)):
                labeled_span = query_labeled_spans[ei]
                offset = query_offset_mapping[ei]
                new_labeled_span = list()
                # starts, ends = feature['start'], feature['end']
                # print('starts=', starts)
                # print('ends=', ends)
                position_map = {}
                for i, (m, n) in enumerate(offset):
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i

                for span in labeled_span:
                    start, end = span
                    end -= 1
                    # if start == 0:
                    #     # assert end == -1
                    #     labels[ei, 0, 0, 0] = 1
                    #     new_labeled_span.append([0, 0])

                    if start in position_map and end in position_map:
                        new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_spans.append(new_labeled_span)

            pred_spans, pred_spans_ = results[id_]["spans"], list()
            pred_types, pred_types_ = results[id_]["types"], list()

            predictions[id_] = {
                'labeled_spans': new_labeled_spans,
                'labeled_types': query_labeled_types,
                'predicted_spans': pred_spans,
                'predicted_types': pred_types,
                'num_class': num_class
            }

        return predictions

    def compute_metrics(self, eval_predictions, stage="dev"):
        '''
        eval_predictions: huggingface
        eval_predictions[0]: logits
        eval_predictions[1]: labels
        # print("raw_datasets=", raw_datasets["validation"])
            Dataset({
                features: ['id', 'support_labeled_spans', 'support_labeled_types', 'support_sentence_num', 'query_labeled_spans', 'query_labeled_types', 'query_sentence_num', 'stage', 'support_input', 'query_input'],
                num_rows: 1000
            })
        '''
        all_metrics = {
            "span_precision": 0.,
            "span_recall": 0.,
            "eval_span_f1": 0,
            "class_precision": 0.,
            "class_recall": 0.,
            "eval_class_f1": 0,
        }

        examples = self.raw_datasets['validation'] if stage == "dev" else self.raw_datasets['test']
        # golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions = self.get_predict_result(eval_predictions[0], examples, stage)

        # === copy from Few-NERD metric ===

        pred_span_cnt = 0  # pred entity cnt
        label_span_cnt = 0  # true label entity cnt
        correct_span_cnt = 0  # correct predicted entity cnt

        pred_class_cnt = 0  # pred entity cnt
        label_class_cnt = 0  # true label entity cnt
        correct_class_cnt = 0  # correct predicted entity cnt

        for episode_id, predicts in predictions.items():
            query_labeled_spans = predicts['labeled_spans']
            query_labeled_types = predicts['labeled_types']
            pred_span = predicts['predicted_spans']
            pred_type = predicts['predicted_types']
            num_class = predicts['num_class']
            for label_span, label_type, pred_span, pred_type in zip(
                    query_labeled_spans, query_labeled_types, pred_span, pred_type
            ):
                label_span_dict = {0: list()}
                pred_span_dict = {0: list()}
                label_class_dict = dict()
                pred_class_dict = dict()
                for span, type in zip(label_span, label_type):
                    label_span_dict[0].append((span[0], span[1]))
                    if type not in label_class_dict.keys():
                        label_class_dict[type] = list()
                    label_class_dict[type].append((span[0], span[1]))

                for span, type in zip(pred_span, pred_type):
                    pred_span_dict[0].append((span[0], span[1]))
                    if type == num_class or span == [0, 0]:
                        continue
                    if type not in pred_class_dict.keys():
                        pred_class_dict[type] = list()
                    pred_class_dict[type].append((span[0], span[1]))

                tmp_pred_span_cnt, tmp_label_span_cnt, correct_span = self.metrics_by_entity(
                    label_span_dict, pred_span_dict
                )

                tmp_pred_class_cnt, tmp_label_class_cnt, correct_class = self.metrics_by_entity(
                    label_class_dict, pred_class_dict
                )
                pred_span_cnt += tmp_pred_span_cnt
                label_span_cnt += tmp_label_span_cnt
                correct_span_cnt += correct_span

                pred_class_cnt += tmp_pred_class_cnt
                label_class_cnt += tmp_label_class_cnt
                correct_class_cnt += correct_class

        span_precision = correct_span_cnt / pred_span_cnt
        span_recall = correct_span_cnt / label_span_cnt
        try:
            span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall)
        except:
            span_f1 = 0.

        class_precision = correct_class_cnt / pred_class_cnt
        class_recall = correct_class_cnt / label_class_cnt
        try:
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)
        except:
            class_f1 = 0.
        all_metrics['span_precision'], all_metrics['span_recall'], all_metrics['eval_span_f1'] = \
            span_precision, span_recall, span_f1
        all_metrics['class_precision'], all_metrics['class_recall'], all_metrics['eval_class_f1'] = \
            class_precision, class_recall, class_f1
        if stage == "dev":
            print("[development dataset] all_metrics=", all_metrics)
        else:
            # print("[testing dataset] all_metrics=", all_metrics)
            print("**** Testing Result ****")
            for key, value in all_metrics.items():
                print("{}:".format(key), value)
        return all_metrics

    def metrics_by_entity(self, label_class_span, pred_class_span):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''

        # pred_class_span # {label:[(start_pos, end_pos), ...]}
        # label_class_span # {label:[(start_pos, end_pos), ...]}
        def get_cnt(label_class_span):
            '''
            return the count of entities
            '''
            cnt = 0
            for label in label_class_span:
                cnt += len(label_class_span[label])
            return cnt

        def get_intersect_by_entity(pred_class_span, label_class_span):
            '''
            return the count of correct entity
            '''
            cnt = 0
            for label in label_class_span:
                cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
            return cnt

        pred_cnt = get_cnt(pred_class_span)
        label_cnt = get_cnt(label_class_span)
        correct_cnt = get_intersect_by_entity(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def save_result(self, logits, label_ids):
        self.compute_metrics((logits,), stage='test')
