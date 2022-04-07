# coding=utf-8
# Copyright (c) 2021 Alibaba PAI team.
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

import pathlib
import random
import re
from shutil import copy
from typing import Iterable, List, Mapping, Tuple, Type

import torch
from easynlp.appzoo.application import Application

from ...core.predictor import Predictor
from ...modelzoo import AutoTokenizer
from ...utils import get_pretrain_model_path, io


class DataAugmentationPredictor(Predictor):
    """ Used to pre-process data (random masking), predict, fill in the blanks, and finally output
    the augmented data in format. """
    def __init__(
        self,
        model_dir: str,
        model_cls: Type[Application] = None,
        user_defined_parameters: dict = None,
        **kwargs,
    ):
        if model_dir.startswith('oss//'):
            local_dir = pathlib.Path('~/.cache').expanduser() / model_dir.split('/')[-1]
            local_dir.mkdir(parents=True, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir

        self.model_dir = model_dir

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        pretrained_model_name_or_path = user_defined_parameters.get(
            'pretrain_model_name_or_path', None
        )
        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
            self.model = model_cls(pretrained_model_name_or_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        else:
            self.model = model_cls.from_pretrained(model_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.first_sequence = kwargs.pop('first_sequence', 'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.sequence_length = kwargs.pop('sequence_length', 128)

        self.output_file = kwargs['output_file']
        self.input_schema = kwargs['input_schema']

        self.da_args = user_defined_parameters['app_parameters']
        self.mlm_mask_prop = self.da_args['mask_proportion']
        self.expansion_rate = self.da_args['expansion_rate']

        self.remove_blanks = self.da_args.get('remove_blanks', False)
        self.blank_re = re.compile(r'([^a-zA-Z])\s+(?=[^a-zA-z])') if self.remove_blanks else None

    def _whole_word_mask(self, input_ids, max_predictions=128):
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        cand_indices = list()

        for i, token in enumerate(input_tokens):
            if token in [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
            ]:
                continue

            if len(cand_indices) > 0 and token.startswith('##'):
                cand_indices[-1].append(i)
            else:
                cand_indices.append([i])

        random.shuffle(cand_indices)
        num_to_predict = min(max(round(len(cand_indices) * self.mlm_mask_prop), 1), max_predictions)
        masked_lms = list()
        covered_indices = set()

        for ids_set in cand_indices:
            if len(masked_lms) >= num_to_predict:
                break
            if len(masked_lms) + len(ids_set) > num_to_predict:
                continue
            if any(index in covered_indices for index in ids_set):
                continue
            for index in ids_set:
                covered_indices.add(index)
                masked_lms.append(index)

        assert len(covered_indices) == len(masked_lms)

        mask_labels = [1 if index in covered_indices else 0 for index in range(len(input_tokens))]
        mask_span_indices = [t for t in cand_indices if t[0] in covered_indices]

        return mask_labels, mask_span_indices

    def _mask_tokens(self, inputs: List[int],
                     mask_labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs_t = torch.LongTensor(inputs)
        mask_labels_t = torch.BoolTensor(mask_labels)
        labels_t = inputs_t.clone()
        labels_t[~mask_labels_t] = -100
        inputs_t[mask_labels_t] = self.tokenizer.mask_token_id

        return inputs_t, labels_t

    def preprocess(self, in_data: List[Mapping]):
        if in_data is None:
            raise ValueError('Input data should not be none.')
        assert isinstance(in_data, Iterable)

        batch_attention_mask = []
        batch_input_ids = []
        batch_label_ids = []
        batch_mask_span_indices = []
        batch_other_cols = []

        for record in in_data:
            sent1 = record.pop(self.first_sequence)
            sent2 = record.pop(self.second_sequence) if self.second_sequence else None
            feature = self.tokenizer(
                sent1,
                sent2,
                padding='max_length',
                truncation=True,
                max_length=self.sequence_length,
            )

            for _ in range(self.expansion_rate):
                mask_labels, mask_span_indices = self._whole_word_mask(feature['input_ids'])
                input_ids, label_ids = self._mask_tokens(feature['input_ids'], mask_labels)

                batch_input_ids.append(input_ids)
                batch_label_ids.append(label_ids)
                batch_attention_mask.append(torch.LongTensor(feature['attention_mask']))
                batch_mask_span_indices.append(mask_span_indices)
                batch_other_cols.append(record)

        assert len(batch_input_ids) == len(batch_label_ids) \
            == len(batch_attention_mask) == len(batch_other_cols)

        return {
            'input_ids': torch.stack(batch_input_ids).to(self.device),
            'attention_mask': torch.stack(batch_attention_mask).to(self.device)
        }, batch_mask_span_indices, batch_other_cols

    def predict(self, in_data: dict):
        inputs, *others = in_data
        with torch.no_grad():
            return self.model(inputs), inputs['input_ids'], others

    def postprocess(self, result):
        outputs, input_ids, others = result
        mask_span_indices, other_cols = others
        logits = outputs['logits']
        input_ids = input_ids.tolist()
        predictions = logits.argmax(dim=-1).tolist()

        # Fill in the predicted results in the masked positions.
        for pred, ids, mask in zip(predictions, input_ids, mask_span_indices):
            for word_set in mask:
                for pos in word_set:
                    ids[pos] = pred[pos]

        batch_lines = list()
        for ids, oc in zip(input_ids, other_cols):
            special_ids = [self.tokenizer.cls_token_id, self.tokenizer.pad_token_id]
            cleaned_ids = [id for id in ids if id not in special_ids]

            sep_token = self.tokenizer.sep_token
            source = self.tokenizer.decode(cleaned_ids).split(sep_token)
            source = [subsq.strip() for subsq in source]
            line = list()
            for field in self.input_schema.split(','):
                field = field.split(':')[0]
                if field == self.first_sequence:
                    sent = self.blank_re.sub(r'\1', source[0]) if self.blank_re else source[0]
                    line.append(sent)
                elif field == self.second_sequence:
                    sent = self.blank_re.sub(r'\1', source[1]) if self.blank_re else source[1]
                    line.append(sent)
                else:
                    line.append(str(oc[field]))
            strline = '\t'.join(line)

            # Preliminary support for BERT tokenizer
            strline = strline.replace(' ##', '')
            strline = strline.replace('##', '')

            batch_lines.append(strline)

        output_dict_list = [dict(augmented_data='\n'.join(batch_lines))]

        return output_dict_list
