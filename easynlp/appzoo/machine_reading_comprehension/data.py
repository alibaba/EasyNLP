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
import torch
import numpy as np
import json
import logging
import collections

from ...modelzoo import AutoTokenizer,BertTokenizer
from ...utils import io
from ..dataset import BaseDataset

logger = logging.getLogger(__name__)



class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        language,
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        self.language = language

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if language == 'zh':        # 中文直接append
                doc_tokens.append(c)
            else:                       # 英文需要按空格切分预处理
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            start_position_character = int(start_position_character)
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


def _is_whitespace(c):
    # if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    if c == " " or c == "\t" or c == "\r" or ord(c) == 0x202F:
        return True
    return False



class InputFeatures(object):
    def __init__(self,
                 unique_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None
                 ):
        self.unique_id = unique_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible



def _check_is_max_context(doc_spans, cur_span_index, position):
    """计算每个token在每一段滑窗中的最佳位置
    maximum context分数：上下文最小值"""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


def convert_example_to_features(
        example,
        tokenizer,
        max_query_length,
        is_training,
        sequence_length,
        doc_stride
):
    """问题若超过max_query_length则会截断取前半部分，
    文档若超过sequence_length则会使用滑窗法"""

    features = []

    unique_id = example.qas_id

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    # 下面这段主要针对英文，有前缀、后缀，中文则会去掉空格
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    # if is_training and example.is_impossible:
    if example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1

    # if is_training and not example.is_impossible:
    if not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

    # print("doc_tokens: ", example.doc_tokens)
    # print("tok_to_orig_index: ", tok_to_orig_index)
    # print("orig_to_tok_index: ", orig_to_tok_index)
    # print("all_doc_tokens: ", all_doc_tokens)
    # print("example.char_to_word_offset: ", example.char_to_word_offset)
    # print("example.start_position: ", example.start_position)
    # print("example.end_position: ", example.end_position)
    # print("tok_start_position: ", tok_start_position)
    # print("tok_end_position: ", tok_end_position)

    max_tokens_for_doc = sequence_length - len(query_tokens) - 3  # 3:[CLS],[SEP],[SEP]

    # 滑窗法
    _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)

            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # zero-pad up to the sequence length
        input_mask = [1] * len(input_ids)

        while len(input_ids) < sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == sequence_length
        assert len(input_mask) == sequence_length
        assert len(segment_ids) == sequence_length

        start_position = 0
        end_position = 0
        # if is_training and not example.is_impossible:
        if not example.is_impossible:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            # query是否在doc_span中
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = sequence_length
                end_position = sequence_length
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        # print("unique_id: ", unique_id)
        # print("doc_span_index: ", doc_span_index)
        # print("tokens: ", str(tokens))
        # print("token_to_orig_map: ", str(token_to_orig_map))
        # print("token_is_max_context: ", str(token_is_max_context))
        # print("input_ids: ", str(input_ids))
        # print("input_mask: ", str(input_mask))
        # print("segment_ids: ", str(segment_ids))
        # print("start_position: ", str(start_position))
        # print("end_position: ", str(end_position))

        features.append(InputFeatures(unique_id=unique_id,
                                      doc_span_index=doc_span_index,
                                      tokens=tokens,
                                      token_to_orig_map=token_to_orig_map,
                                      token_is_max_context=token_is_max_context,
                                      input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      start_position=start_position,
                                      end_position=end_position,
                                      is_impossible=example.is_impossible))

    return features



class MachineReadingComprehensionDataset(BaseDataset):
    """
    MachineReadingComprehension Dataset

    Args:
        data_file: input data file.
        pretrained_model_name_or_path: for init tokenizer.
        sequence_length: max sequence length of each input instance.
        first_sequence: input query text
        second_sequence: input context text
        answer: input answer text
    """
    def __init__(self,
                 data_file,
                 pretrained_model_name_or_path,
                 sequence_length,
                 input_schema,
                 first_sequence,
                 second_sequence,
                 answer_name,
                 qas_id,
                 start_position_name,
                 max_query_length,
                 doc_stride,
                 user_defined_parameters,
                 is_training,
                 *args,
                 **kwargs):

        super(MachineReadingComprehensionDataset, self).__init__(data_file,
                                                                 input_schema=input_schema,
                                                                 output_format="dict",
                                                                 *args,
                                                                 **kwargs)

        ### user_defined_parameters
        if user_defined_parameters is not None:
            if type(user_defined_parameters) == 'str':
                self.user_defined_parameters = json.loads(user_defined_parameters)
            else:
                self.user_defined_parameters = user_defined_parameters
        else:
            self.user_defined_parameters = {}

        ### tokenizer
        if os.path.exists(pretrained_model_name_or_path):
            local_path = pretrained_model_name_or_path
        else:
            local_path = os.environ['HOME'] + '/.easynlp/modelzoo/' + pretrained_model_name_or_path
        # self.tokenizer = BertTokenizer(vocab_file=local_path + '/vocab.txt', sep_token="[SEP]",
        #                                pad_token="[PAD]", cls_token="[CLS]")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        self.is_training = is_training
        self.sequence_length = sequence_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.language = self.user_defined_parameters.get("language", 'zh')

        # Text Features
        assert first_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % first_sequence
        assert second_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % second_sequence
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.answer_name = answer_name
        self.qas_id = qas_id
        self.start_position_name = start_position_name


    def convert_single_row_to_example(self, row):

        question_text = row[self.first_sequence]
        context_text = row[self.second_sequence]
        answer_text = row[self.answer_name] if self.answer_name else None
        qas_id = row[self.qas_id] if self.qas_id else None
        start_position_character = row[self.start_position_name] if self.start_position_name else None
        language = self.language

        # print("qas_id: ", qas_id)
        # print("question_text: ", question_text)
        # print("context_text: ", context_text)
        # print("answer_text: ", answer_text)
        # print("start_position_character: ", start_position_character)
        # print("language: ", language)

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            context_text=context_text,
            answer_text=answer_text,
            start_position_character=start_position_character,
            language=language
        )

        return convert_example_to_features(example,
                                           self.tokenizer,
                                           self.max_query_length,
                                           self.is_training,
                                           self.sequence_length,
                                           self.doc_stride)



    def batch_fn(self, features):
        """
        Args:
            features (`list`): a list of features produced by `convert_single_row_to_example`
        Returns:
            inputs (`dict`): a dict to model forwarding
        """

        unique_id_list = []
        tokens_list = []
        tok_to_orig_index_list = []
        token_is_max_context_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        start_position_list = []
        end_position_list = []
        label_ids_list = []

        for feature in features:
            for f in feature:
                unique_id_list.append(f.unique_id)
                tokens_list.append(f.tokens)
                tok_to_orig_index_list.append(f.token_to_orig_map)
                token_is_max_context_list.append(f.token_is_max_context)
                input_ids_list.append(f.input_ids)
                attention_mask_list.append(f.input_mask)
                token_type_ids_list.append(f.segment_ids)
                start_position_list.append(f.start_position)
                end_position_list.append(f.end_position)
                label_ids_list.append([f.start_position, f.end_position])

        inputs = {
            "unique_id": unique_id_list,
            "tokens": tokens_list,
            "tok_to_orig_index": tok_to_orig_index_list,
            "token_is_max_context": token_is_max_context_list,
            "input_ids": torch.LongTensor(input_ids_list),
            "attention_mask": torch.LongTensor(attention_mask_list),
            "token_type_ids": torch.LongTensor(token_type_ids_list),
            "start_position": torch.LongTensor(start_position_list),
            "end_position": torch.LongTensor(end_position_list),
            "label_ids": torch.LongTensor(label_ids_list)
        }

        return inputs

