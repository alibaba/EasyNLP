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

import json
import os
import traceback
import uuid
from threading import Lock

import numpy as np
import torch
from scipy.special import log_softmax
from torch.utils.data import DataLoader

from easynlp.core.predictor import (Predictor, PyModelPredictor,
                                    get_model_predictor)
from easynlp.fewshot_learning.fewshot_dataset import FewshotBaseDataset
from easynlp.modelzoo import AutoTokenizer
from easynlp.utils import io
from easynlp.utils.global_vars import parse_user_defined_parameters


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class FewshotPyModelPredictor(PyModelPredictor):
    """This is the predictor class for supporting fewshot learning.

    Args:
        model_cls:
            The classification model of the pretrained model for fewshot classification.
        saved_model_path:
            The path of the saved model.
        input_keys:
            The collection of input keys for prediction.
        user_defined_parameters:
            The dict of user defined parameters for fewshot prediction.
        output_keys:
            The collection of output keys for prediction.
    """
    def __init__(self, model_cls, saved_model_path, input_keys,
                 user_defined_parameters, output_keys, **kwargs):
        self.model = model_cls.from_pretrained(
            pretrained_model_name_or_path=saved_model_path,
            user_defined_parameters=user_defined_parameters,
            **kwargs)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.input_keys = input_keys
        self.output_keys = output_keys

    def predict(self, in_data):
        """The main predictor function for fewshot learning.

        Args:
            in_data: the dict of input data

        Returns: the dict of output tensors containing the results
        """
        in_tensor = dict()
        for key, tensor_type in self.input_keys:
            if tensor_type is None or isinstance(in_data[key], tensor_type):
                in_tensor[key] = in_data[key]
            else:
                in_tensor[key] = tensor_type(in_data[key])
            if torch.cuda.is_available() and tensor_type is not None:
                in_tensor[key] = in_tensor[key].cuda()
        with torch.no_grad():
            predictions = self.model.forward({
                'input_ids':
                in_tensor['input_ids'],
                'attention_mask':
                in_tensor['attention_mask']
            })
        ret = {}
        for key, val in in_tensor.items():
            ret[key] = val
        for key in self.output_keys:
            ret[key] = predictions[key]
        return ret


class PromptPredictor(Predictor):
    """This is the predictor class for supporting prompt-based fewshot learning.

    Args:
        model_dir:
            The path of the model
        model_cls:
            The classification model of the pretrained model for fewshot classification.
        user_defined_parameters:
            The dict of user defined parameters for fewshot prediction.
    """
    def __init__(self,
                 model_dir,
                 model_cls=None,
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super(PromptPredictor, self).__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model_predictor = FewshotPyModelPredictor(
            saved_model_path=model_dir,
            model_cls=model_cls,
            user_defined_parameters=user_defined_parameters,
            input_keys=[('input_ids', torch.LongTensor),
                        ('attention_mask', torch.LongTensor),
                        ('label_ids', torch.LongTensor),
                        ('mask_span_indices', None)],
            output_keys=['logits'])
        self.label_path = os.path.join(model_dir, 'label_mapping.json')
        self.MUTEX = Lock()
        try:
            user_defined_parameters_dict = user_defined_parameters.get(
                'app_parameters')
        except KeyError:
            traceback.print_exc()
            exit(-1)
        pattern = user_defined_parameters_dict.get('pattern')
        label_desc = user_defined_parameters_dict.get('label_desc')
        with io.open(self.label_path) as f:
            self.label_mapping = json.load(f)
        self.label_id_to_name = {
            idx: name
            for name, idx in self.label_mapping.items()
        }
        self.first_sequence = kwargs.pop('first_sequence', 'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.label_name = kwargs.pop('label_name', 'label')
        self.sequence_length = kwargs.pop('sequence_length', 128)
        self.label_map = dict(
            zip(self.label_mapping.keys(), label_desc.split(',')))
        self.pad_idx = self.tokenizer.pad_token_id
        self.mask_idx = self.tokenizer.mask_token_id
        assert pattern is not None, 'You must define the pattern for PET learning'
        pattern_list = pattern.split(',')
        assert self.first_sequence in pattern_list and (not self.second_sequence or self.second_sequence in pattern_list)\
            , 'All text columns should be included in the pattern'

        cnt = 0
        for i in range(len(pattern_list)):
            if pattern_list[i] == '<pseudo>':
                pattern_list[i] = '<pseudo-%d>' % cnt
                cnt += 1
        if cnt > 0:
            self.tokenizer.add_tokens(['<pseudo-%d>' % i for i in range(cnt)])
        try:
            self.MUTEX.acquire()
            self.pattern = [
                self.tokenizer.tokenize(s) if s not in (self.first_sequence,
                                                        self.second_sequence,
                                                        self.label_name) else s
                for s in pattern_list
            ]
        finally:
            self.MUTEX.release()
        label_desc = user_defined_parameters_dict.get('label_desc')
        if not label_desc:
            print(
                'Using Contrastive Few shot Learner, using random label words only as place-holders'
            )
            label_desc = [s[0] for s in self.label_mapping.keys()]
        else:
            label_desc = label_desc.split(',')
        self.masked_length = len(self.tokenizer.tokenize(label_desc[0]))
        self.label_map = dict(zip(self.label_mapping.keys(), label_desc))
        self.num_extra_tokens = sum(
            [len(s) if isinstance(s, list) else 0
             for s in self.pattern]) + self.masked_length

        def str_to_ids(s, tokenizer):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

        self.label_candidates_ids = {
            k: str_to_ids(v, self.tokenizer)
            for k, v in self.label_map.items()
        }

    def preprocess(self, in_data):
        """The preprocess step of model prediction.

        Args:
            in_data: the dict of input data

        Returns: the dict of output tensors containing the results
        """

        if not in_data:
            raise RuntimeError('Input data should not be None.')

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'label_ids': [],
            'mask_span_indices': []
        }

        for record in in_data:
            text_a = record[self.first_sequence]
            text_b = record.get(self.second_sequence, None)
            try:
                self.MUTEX.acquire()
                tokens_a = self.tokenizer.tokenize(text_a)
            finally:
                self.MUTEX.release()
            max_seq_length = self.sequence_length
            max_seq_length -= self.num_extra_tokens
            tokens_b = None
            if text_b:
                try:
                    self.MUTEX.acquire()
                    tokens_b = self.tokenizer.tokenize(text_b)
                finally:
                    self.MUTEX.release()
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)
            else:
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
            # Prediction mode
            label = self.tokenizer.mask_token * self.masked_length
            try:
                self.MUTEX.acquire()
                label_tokens = self.tokenizer.tokenize(label)
            finally:
                self.MUTEX.release()
            assert len(
                label_tokens
            ) == self.masked_length, 'label length should be equal to the mask length'
            tokens = [self.tokenizer.cls_token]
            label_position = None
            for p in self.pattern:
                if p == self.first_sequence:
                    tokens += tokens_a
                elif p == self.second_sequence:
                    tokens += (tokens_b if tokens_b else [])
                elif p == self.label_name:
                    label_position = len(tokens)
                    tokens += label_tokens
                elif isinstance(p, list):
                    tokens += p
                else:
                    raise ValueError('Unexpected pattern---' + p)
            try:
                self.MUTEX.acquire()
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            finally:
                self.MUTEX.release()
            length = len(input_ids)
            attention_mask = [1] * length
            token_type_ids = [0] * length
            mask_labels = [-100] * length
            mask_span_indices = []
            for i in range(self.masked_length):
                mask_labels[label_position + i] = 1
                mask_span_indices.append([label_position + i])
            max_seq_length += self.num_extra_tokens
            # token padding
            input_ids += [
                self.pad_idx,
            ] * (max_seq_length - length)
            attention_mask += [
                self.pad_idx,
            ] * (max_seq_length - length)
            token_type_ids += [0] * (max_seq_length - length)
            mask_labels += [-100] * (max_seq_length - length)

            rst['input_ids'].append(input_ids)
            rst['attention_mask'].append(attention_mask)
            rst['token_type_ids'].append(token_type_ids)
            rst['label_ids'].append(mask_labels)
            rst['mask_span_indices'].append(mask_span_indices)

        return rst

    def predict(self, in_data):
        """The main function calling the predictor."""
        return self.model_predictor.predict(in_data)

    def postprocess(self, result):
        """The  postprocess that converts logits to final results.

        Args:
            result: the dict of prediction results by the model

        Returns: the list of the final results.
        """

        logits = result['logits']
        predictions = []
        label_ids = result.pop('label_ids')
        mask_span_indices = result.pop('mask_span_indices')

        for b in range(logits.shape[0]):
            _logits = logits[b]
            indices = mask_span_indices[b]
            y_pred = torch.nn.functional.log_softmax(_logits)
            preds = []
            for k, v in self.label_candidates_ids.items():
                pred_prob = 0.
                for l_ids, span_indices in zip(v, indices):
                    span_idx = span_indices[0]
                    pred_prob += y_pred[span_idx, l_ids].item()
                preds.append({'pred': k, 'log_probability': pred_prob})
            preds.sort(key=lambda x: -x['log_probability'])
            predictions.append(preds)

        new_results = list()
        for b, preds in enumerate(predictions):
            new_result = list()
            for pred in preds:
                new_result.append({
                    'pred': pred['pred'],
                    'log_probability': pred['log_probability'],
                })
            new_results.append({
                'id':
                result['id'][b] if 'id' in result else str(uuid.uuid4()),
                'output':
                new_result,
                'predictions':
                new_result[0]['pred'],
            })
        if len(new_results) == 1:
            new_results = new_results[0]
        return new_results


class CPTPredictor(Predictor):
    """This is the predictor class for supporting CPT fewshot learning.

    Args:
        model_dir:
            The path of the model
        model_cls:
            The classification model of the pretrained model for fewshot classification.
        user_defined_parameters:
            The dict of user defined parameters for fewshot prediction.
    """
    def __init__(self,
                 model_dir,
                 model_cls=None,
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super(CPTPredictor, self).__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.MUTEX = Lock()
        self.model_predictor = FewshotPyModelPredictor(
            saved_model_path=model_dir,
            model_cls=model_cls,
            user_defined_parameters=user_defined_parameters,
            input_keys=[('input_ids', torch.LongTensor),
                        ('attention_mask', torch.LongTensor),
                        ('label_ids', torch.LongTensor),
                        ('mask_span_indices', None)],
            output_keys=['features'])
        self.label_path = os.path.join(model_dir, 'label_mapping.json')
        try:
            user_defined_parameters_dict = user_defined_parameters.get(
                'app_parameters')
        except KeyError:
            traceback.print_exc()
            exit(-1)
        pattern = user_defined_parameters_dict.get('pattern')
        label_desc = user_defined_parameters_dict.get('label_desc')
        with io.open(self.label_path) as f:
            self.label_mapping = json.load(f)
        self.label_id_to_name = {
            idx: name
            for name, idx in self.label_mapping.items()
        }
        self.first_sequence = kwargs.pop('first_sequence', 'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.label_name = kwargs.pop('label_name', 'label')
        self.sequence_length = kwargs.pop('sequence_length', 128)
        self.pad_idx = self.tokenizer.pad_token_id
        self.mask_idx = self.tokenizer.mask_token_id
        assert pattern is not None, 'You must define the pattern for PET learning'
        pattern_list = pattern.split(',')
        assert self.first_sequence in pattern_list and (not self.second_sequence or self.second_sequence in pattern_list)\
            , 'All text columns should be included in the pattern'
        cnt = 0
        for i in range(len(pattern_list)):
            if pattern_list[i] == '<pseudo>':
                pattern_list[i] = '<pseudo-%d>' % cnt
                cnt += 1
        if cnt > 0:
            self.tokenizer.add_tokens(['<pseudo-%d>' % i for i in range(cnt)])

        try:
            self.MUTEX.acquire()
            self.pattern = [
                self.tokenizer.tokenize(s) if s not in (self.first_sequence,
                                                        self.second_sequence,
                                                        self.label_name) else s
                for s in pattern_list
            ]
        finally:
            self.MUTEX.release()

        label_desc = user_defined_parameters_dict.get('label_desc')
        if not label_desc:
            print('CPT: using random label words as place-holders')
            label_desc = [s[0] for s in self.label_mapping.keys()]
        else:
            label_desc = label_desc.split(',')
        try:
            self.MUTEX.acquire()
            self.masked_length = len(self.tokenizer.tokenize(label_desc[0]))
        finally:
            self.MUTEX.release()
        self.label_map = dict(zip(self.label_mapping.keys(), label_desc))
        self.num_extra_tokens = sum(
            [len(s) if isinstance(s, list) else 0
             for s in self.pattern]) + self.masked_length

        def str_to_ids(s, tokenizer):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

        self.label_candidates_ids = {
            str_to_ids(v, self.tokenizer)[0]: k
            for k, v in self.label_map.items()
        }

        anchor_args = kwargs.pop('few_shot_anchor_args', None)
        assert anchor_args is not None

        anchor_data = FewshotBaseDataset(
            data_file=anchor_args.tables.split(',')[0],
            pretrained_model_name_or_path=anchor_args.
            pretrained_model_name_or_path,
            max_seq_length=anchor_args.sequence_length,
            first_sequence=anchor_args.first_sequence,
            second_sequence=anchor_args.second_sequence,
            label_name=anchor_args.label_name,
            label_enumerate_values=anchor_args.label_enumerate_values,
            user_defined_parameters=parse_user_defined_parameters(
                anchor_args.user_defined_parameters),
            is_training=True,
            *args,
            **kwargs)
        anchor_batch_size = kwargs.pop('anchor_batch_size', 16)
        assert anchor_data is not None

        self.train_dataloader = DataLoader(anchor_data,
                                           batch_size=anchor_batch_size,
                                           shuffle=False,
                                           collate_fn=anchor_data.batch_fn)
        anchor_list = []
        anchor_labels = []
        for _, batch in enumerate(self.train_dataloader):
            with torch.no_grad():
                outputs = self.model_predictor.predict(batch)
                label_ids = outputs['label_ids']
                labels = label_ids[label_ids > 0].detach().cpu().numpy()
                features = torch.nn.functional.normalize(
                    outputs['features'][label_ids > 0]).detach().cpu().numpy()
                anchor_list.append(features)
                anchor_labels.append(labels)
        self.anchor_feature = np.concatenate(anchor_list)
        self.anchor_labels = np.concatenate(anchor_labels)

    def preprocess(self, in_data):
        """The preprocess step of model prediction.

        Args:
            in_data: the dict of input data

        Returns: the dict of output tensors containing the results
        """

        if not in_data:
            raise RuntimeError('Input data should not be None.')

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'label_ids': [],
            'mask_span_indices': []
        }

        for record in in_data:
            text_a = record[self.first_sequence]
            text_b = record.get(self.second_sequence, None)
            try:
                self.MUTEX.acquire()
                tokens_a = self.tokenizer.tokenize(text_a)
            finally:
                self.MUTEX.release()
            max_seq_length = self.sequence_length
            max_seq_length -= self.num_extra_tokens
            tokens_b = None
            if text_b:
                try:
                    self.MUTEX.acquire()
                    tokens_b = self.tokenizer.tokenize(text_b)
                finally:
                    self.MUTEX.release()
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)
            else:
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]
            # Prediction mode
            label = self.tokenizer.mask_token * self.masked_length
            try:
                self.MUTEX.acquire()
                label_tokens = self.tokenizer.tokenize(label)
            finally:
                self.MUTEX.release()
            assert len(
                label_tokens
            ) == self.masked_length, 'label length should be equal to the mask length'
            tokens = [self.tokenizer.cls_token]
            label_position = None
            for p in self.pattern:
                if p == self.first_sequence:
                    tokens += tokens_a
                elif p == self.second_sequence:
                    tokens += (tokens_b if tokens_b else [])
                elif p == self.label_name:
                    label_position = len(tokens)
                    tokens += label_tokens
                elif isinstance(p, list):
                    tokens += p
                else:
                    raise ValueError('Unexpected pattern---' + p)
            try:
                self.MUTEX.acquire()
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            finally:
                self.MUTEX.release()
            length = len(input_ids)
            attention_mask = [1] * length
            token_type_ids = [0] * length
            mask_labels = [-100] * length
            mask_span_indices = []
            for i in range(self.masked_length):
                mask_labels[label_position + i] = 1
                mask_span_indices.append([label_position + i])
            max_seq_length += self.num_extra_tokens
            # token padding
            input_ids += [
                self.pad_idx,
            ] * (max_seq_length - length)
            attention_mask += [
                self.pad_idx,
            ] * (max_seq_length - length)
            token_type_ids += [0] * (max_seq_length - length)
            mask_labels += [-100] * (max_seq_length - length)

            rst['input_ids'].append(input_ids)
            rst['attention_mask'].append(attention_mask)
            rst['token_type_ids'].append(token_type_ids)
            rst['label_ids'].append(mask_labels)
            rst['mask_span_indices'].append(mask_span_indices)

        return rst

    def predict(self, in_data):
        """The main function calling the predictor."""
        return self.model_predictor.predict(in_data)

    def postprocess(self, result):
        """The postprocess that converts embeddings of CPT to final results.

        Args:
            result: the dict of prediction results by the model

        Returns: the list of the final results.
        """

        predictions = []
        with torch.no_grad():
            label_ids = result.pop('label_ids')
            labels = label_ids[label_ids > 0].detach().cpu().numpy()
            features = torch.nn.functional.normalize(
                result['features'][label_ids > 0]).detach().cpu().numpy()
            dist = np.dot(features, self.anchor_feature.T)
            pred = self.anchor_labels[np.argmax(dist, -1)]
        predictions.extend(pred)

        new_results = list()
        for b, pred in enumerate(predictions):
            new_results.append({
                'id':
                result['id'][b] if 'id' in result else str(uuid.uuid4()),
                'output':
                self.label_candidates_ids[pred],
                'predictions':
                self.label_candidates_ids[pred],
            })
        if len(new_results) == 1:
            new_results = new_results[0]
        return new_results
