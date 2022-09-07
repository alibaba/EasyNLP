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
import logging
import collections
import uuid
import numpy as np
from threading import Lock
from tqdm import tqdm
from typing import List
import six

import torch

from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer, BertTokenizer
from ...utils import io
from ...fewshot_learning.fewshot_predictor import PromptPredictor, CPTPredictor
from .data import SquadExample, _check_is_max_context


logger = logging.getLogger()


class Vocab:
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):

    return Vocab(vocab_path)


def convert_single_example_to_features(
        example,
        tokenizer,
        max_query_length,
        sequence_length,
        doc_stride
):

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    orig_to_tok_index = []
    all_doc_tokens = []
    # 下面这段主要针对英文，有前缀、后缀，中文则会去掉空格
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            all_doc_tokens.append(sub_token)

    max_tokens_for_doc = sequence_length - len(query_tokens) - 3  # 3:[CLS],[SEP],[SEP]

    # 滑窗法
    _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
    doc_spans = []
    features = []
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

        feature = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'unique_id': example.qas_id,
            'question_text': example.question_text,
            'context_text': example.context_text,
            'tokens': tokens
        }
        features.append(feature)

    return features


class MachineReadingComprehensionPredictor(Predictor):

    def __init__(self, model_dir, model_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = BertTokenizer(vocab_file=model_dir + '/vocab.txt', sep_token="[SEP]",
                                       pad_token="[PAD]", cls_token="[CLS]")

        self.model_predictor = get_model_predictor(
            model_dir=model_dir,
            model_cls=model_cls,
            input_keys=[("input_ids", torch.LongTensor),
                        ("attention_mask", torch.LongTensor),
                        ("token_type_ids", torch.LongTensor)
                        ],
            output_keys=["start_logits", "end_logits", "predictions"])

        self.first_sequence = kwargs.pop("first_sequence", "query")
        self.second_sequence = kwargs.pop("second_sequence", "context")
        self.qas_id = kwargs.pop("qas_id", "qas_id")
        self.sequence_length = kwargs.pop("sequence_length", 384)
        self.max_query_length = kwargs.pop("max_query_length", 64)
        self.max_answer_length = kwargs.pop("max_answer_length", 30)
        self.doc_stride = kwargs.pop("doc_stride", 128)
        self.n_best_size = kwargs.pop("n_best_size", 10)
        self.output_file = kwargs.pop("outputs", "dev.pred.csv")
        self.output_answer_file = kwargs.pop("output_answer_file", "dev.ans.csv")

        self.vocab = build_vocab(vocab_path=model_dir + '/vocab.txt')
        self.MUTEX = Lock()

    def get_format_text_and_word_offset(self, text):
        """
        格式化原始输入的文本（去除多个空格）,同时得到每个字符所属的元素（单词）的位置
        这样，根据原始数据集中所给出的起始index(answer_start)就能立马判定它在列表中的位置。
        :param text:
        :return:
        e.g.
            text = "Architecturally, the school has a Catholic character. "
            return:['Architecturally,', 'the', 'school', 'has', 'a', 'Catholic', 'character.'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3,
             3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
        """

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        # 以下这个for循环的作用就是将原始context中的内容进行格式化
        for c in text:  # 遍历paragraph中的每个字符
            if is_whitespace(c):  # 判断当前字符是否为空格（各类空格）
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:  # 如果前一个字符是空格
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c  # 在list的最后一个元素中继续追加字符
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_offset

    @staticmethod
    def get_token_to_orig_map(input_tokens, origin_context, tokenizer):
        """
           本函数的作用是根据input_tokens和原始的上下文，返回得input_tokens中每个单词在原始单词中所对应的位置索引
           :param input_tokens:  ['[CLS]', 'to', 'whom', 'did', 'the', 'virgin', '[SEP]', 'architectural', '##ly',
                                   ',', 'the', 'school', 'has', 'a', 'catholic', 'character', '.', '[SEP']
           :param origin_context: "Architecturally, the Architecturally, test, Architecturally,
                                    the school has a Catholic character. Welcome moon hotel"
           :param tokenizer:
           :return: {7: 4, 8: 4, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 10}
                   含义是input_tokens[7]为origin_context中的第4个单词 Architecturally,
                        input_tokens[8]为origin_context中的第4个单词 Architecturally,
                        ...
                        input_tokens[10]为origin_context中的第5个单词 the
           """
        # print("orig_map origin_context: ", origin_context)
        origin_context_tokens = origin_context.split()
        # print("orig_map origin_context_tokens: ", origin_context_tokens)
        token_id = []
        str_origin_context = ""
        for i in range(len(origin_context_tokens)):
            # print("orig_map origin_context_tokens[i]: ", origin_context_tokens[i])
            tokens = tokenizer(origin_context_tokens[i])
            # print("orig_map tokens: ", tokens)
            str_token = "".join(tokens)
            # print("orig_map str_token: ", str_token)
            str_origin_context += "" + str_token
            # print("orig_map str_origin_context: ", str_origin_context)
            for _ in str_token:
                token_id.append(i)
        # print("orig_map str_origin_context: ", str_origin_context)

        # print("orig_map input_tokens: ", input_tokens)
        key_start = input_tokens.index('[SEP]') + 1
        tokenized_tokens = input_tokens[key_start:-1]
        str_tokenized_tokens = "".join(tokenized_tokens)
        # print("orig_map str_tokenized_tokens: ", str_tokenized_tokens)
        index = str_origin_context.index(str_tokenized_tokens)
        value_start = token_id[index]
        token_to_orig_map = {}
        # 处理这样的边界情况： Building's gold   《==》   's', 'gold', 'dome'
        token = tokenizer(origin_context_tokens[value_start])
        for i in range(len(token), -1, -1):
            s1 = "".join(token[-i:])
            s2 = "".join(tokenized_tokens[:i])
            if s1 == s2:
                token = token[-i:]
                break

        while True:
            for j in range(len(token)):
                token_to_orig_map[key_start] = value_start
                key_start += 1
                if len(token_to_orig_map) == len(tokenized_tokens):
                    return token_to_orig_map
            value_start += 1
            token = tokenizer(origin_context_tokens[value_start])

    @staticmethod
    def get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        # logits = [0.37203778 0.48594432 0.81051651 0.07998148 0.93529721 0.0476721
        #  0.15275263 0.98202781 0.07813079 0.85410559]
        # n_best_size = 4
        # return [7, 4, 9, 2]
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def get_final_text(self, pred_text, orig_text):
        """Project the tokenized prediction back to the original text."""

        # ref: https://github.com/google-research/bert/blob/master/run_squad.py
        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tok_text = " ".join(self.tokenizer(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def generate_answers(self, result):
        id_to_context_list = collections.defaultdict(list)
        id_to_context_text = collections.defaultdict(list)
        id_to_query = collections.defaultdict(list)
        id_to_input_ids = collections.defaultdict(list)
        id_to_tokens_list = collections.defaultdict(list)
        all_logits_list = collections.defaultdict(list)
        unique_id_list = []
        result_length = len(result["unique_id"])
        # print("generate_answers result_length: ", result_length)

        for i in range(result_length):
            unique_id = result["unique_id"][i]
            # print("generate_answers unique_id: ", unique_id)
            if not (len(unique_id_list) > 0 and unique_id_list[-1] == unique_id):
                unique_id_list.append(unique_id)
            context = result["context_text"][i]
            # print("generate_answers context: ", context)
            context_tokens, word_offset = self.get_format_text_and_word_offset(context)
            # print("generate_answers context_list: ", context_list)
            # print("generate_answers word_offset: ", word_offset)
            id_to_context_list[unique_id].append(context_tokens)
            id_to_context_text[unique_id].append(" ".join(context_tokens))
            id_to_query[unique_id].append(result["question_text"][i])
            id_to_input_ids[unique_id].append(result["input_ids"][i])
            # print("generate_answers input_ids: ", result["input_ids"][i])
            id_to_tokens_list[unique_id].append(result["tokens_list"][i])

            start_logits = result["start_logits"][i]
            # print("generate_answers start_logits_shape: ", np.array(start_logits).shape)
            end_logits = result["end_logits"][i]
            # print("generate_answers end_logits_shape: ", np.array(end_logits).shape)
            all_logits_list[unique_id].append([start_logits, end_logits])
            # print("generate_answers all_logits_list_shape: ", np.array(all_logits_list[unique_id]).shape)
            # print("generate_answers all_logits_list: ", all_logits_list[unique_id])

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["text", "start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = collections.defaultdict(list)

        for unique_id in tqdm(unique_id_list, ncols=80, desc="walking through candidate answers ..."):
            input_ids = id_to_input_ids[unique_id]
            tokens_list = id_to_tokens_list[unique_id]
            all_logits = all_logits_list[unique_id]

            for i in range(len(all_logits)):
                # print("generate_answers all_logits_shape: ", np.array(all_logits).shape)
                # input_ids = id_to_input_ids[unique_id][i]

                # print("generate_answers unique_id start_logits_shape: ", np.array(all_logits[i][0]).shape)
                # print("generate_answers unique_id start_logits: ", np.array(all_logits[i][0]))
                # print("generate_answers unique_id end_logits_shape: ", np.array(all_logits[i][1]).shape)
                # print("generate_answers unique_id end_logits: ", np.array(all_logits[i][1]))
                start_indexes = self.get_best_indexes(all_logits[i][0], self.n_best_size)
                # print("generate_answers unique_id start_indexes_shape: ", np.array(start_indexes).shape)
                # print("generate_answers unique_id start_indexes: ", np.array(start_indexes))
                end_indexes = self.get_best_indexes(all_logits[i][1], self.n_best_size)
                # print("generate_answers unique_id end_indexes_shape: ", np.array(end_indexes).shape)
                # print("generate_answers unique_id end_indexes: ", np.array(end_indexes))

                # print("generate_answers unique_id input_ids[i]_length: ", len(input_ids[i]))
                # print("generate_answers unique_id tokens_list[i]_length: ", len(tokens_list[i]))
                # print("generate_answers unique_id context_text[i]_length: ", len(id_to_context_text[unique_id][i]))
                token_to_orig_map = self.get_token_to_orig_map(tokens_list[i], id_to_context_text[unique_id][i], self.tokenizer.tokenize)
                # print("generate_answers unique_id token_to_orig_map: ", token_to_orig_map)

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # print("generate_answers unique_id start_index: ", start_index)
                        # print("generate_answers unique_id end_index: ", end_index)
                        if start_index >= len(input_ids[i]):
                            # print("start_index >= len(input_ids[i])")
                            continue
                        if end_index >= len(input_ids[i]):
                            # print("end_index >= len(input_ids[i])")
                            continue
                        if start_index not in token_to_orig_map:
                            # print("start_index not in token_to_orig_map")
                            continue
                        if end_index not in token_to_orig_map:
                            # print("end_index not in token_to_orig_map")
                            continue
                        if end_index < start_index:
                            # print("end_index < start_index")
                            continue
                        length = end_index - start_index + 1
                        if length > self.max_answer_length:
                            # print("length > self.max_answer_length")
                            continue
                        token_ids = input_ids[i]
                        # print("generate_answers unique_id token_ids: ", token_ids)
                        strs = [self.vocab.itos[s] for s in token_ids]
                        # print("generate_answers unique_id strs: ", strs)
                        tok_text = " ".join(strs[start_index:(end_index + 1)])
                        # print("generate_answers unique_id tok_text: ", tok_text)
                        tok_text = tok_text.replace(" ##", "").replace("##", "")
                        # print("generate_answers unique_id tok_text: ", tok_text)
                        tok_text = tok_text.strip()
                        tok_text = " ".join(tok_text.split())
                        # print("generate_answers unique_id tok_text: ", tok_text)

                        # print("generate_answers unique_id token_to_orig_map: ", token_to_orig_map)
                        orig_doc_start = token_to_orig_map[start_index]
                        orig_doc_end = token_to_orig_map[end_index]
                        # print("generate_answers unique_id orig_doc_start: ", orig_doc_start)
                        # print("generate_answers unique_id orig_doc_end: ", orig_doc_end)
                        orig_tokens = id_to_context_list[unique_id][i][orig_doc_start:(orig_doc_end + 1)]
                        # print("generate_answers unique_id orig_tokens: ", orig_tokens)
                        orig_text = " ".join(orig_tokens)
                        # print("generate_answers unique_id orig_text: ", orig_text)
                        final_text = self.get_final_text(tok_text, orig_text)
                        # print("generate_answers unique_id final_text: ", final_text)
                        # print("generate_answers unique_id logit_sum: ", all_logits[i][0][start_index] + all_logits[i][1][end_index])

                        # print("generate_answers unique_id all_logits[i][0].shape: ", all_logits[i][0].shape)
                        # print("generate_answers unique_id all_logits[i][1].shape: ", all_logits[i][1].shape)
                        prelim_predictions[unique_id].append(_PrelimPrediction(
                            text=final_text,
                            start_index=int(start_index),
                            end_index=int(end_index),
                            start_logit=float(all_logits[i][0][start_index]),
                            end_logit=float(all_logits[i][1][end_index])))

        for k, v in prelim_predictions.items():
            prelim_predictions[k] = sorted(prelim_predictions[k],
                                           key=lambda x: (x.start_logit + x.end_logit),
                                           reverse=True)

        output_dict_list = []
        best_results = {}
        for k, v in prelim_predictions.items():
            best_results[k] = v[0].text
            output_dict = {
                "unique_id": k,
                "best_answer":  v[0].text,
                "query": id_to_query[k],
                "context": id_to_context_text[k],
                "all_n_best_answer":  "||".join(v_.text for v_ in v)
            }
            output_dict_list.append(output_dict)

        with open(self.output_answer_file, 'w') as f:
            f.write(json.dumps(best_results, indent=4) + '\n')

        return output_dict_list

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "unique_id": [],
            "question_text": [],
            "context_text": [],
            "tokens_list": []
        }

        for record in in_data:
            # print("preprocess in_data_qas_id: ", record["qas_id"])
            # print("preprocess in_data_context_text: ", record["context_text"])
            # print("preprocess in_data_question_text: ", record["question_text"])
            # print("preprocess in_data_answer_text: ", record["answer_text"])
            # print("preprocess in_data_start_position_character: ", record["start_position_character"])

            question_text = record[self.first_sequence]
            context_text = record[self.second_sequence]
            unique_id = record[self.qas_id]

            example = SquadExample(
                qas_id=unique_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=None,
                start_position_character=None
            )
            try:
                self.MUTEX.acquire()
                features = convert_single_example_to_features(example,
                                                              self.tokenizer,
                                                              self.max_query_length,
                                                              self.sequence_length,
                                                              self.doc_stride
                                                              )
            finally:
                self.MUTEX.release()

            for f_index, feature in enumerate(features):
                rst["input_ids"].append(feature["input_ids"])
                rst["attention_mask"].append(feature["attention_mask"])
                rst["token_type_ids"].append(feature["token_type_ids"])
                rst["unique_id"].append(feature["unique_id"])
                rst["question_text"].append(feature["question_text"])
                rst["context_text"].append(feature["context_text"])
                rst["tokens_list"].append(feature["tokens"])

                # print("preprocess f_index: ", f_index)
                # print("preprocess feature_input_ids: ", feature["input_ids"])
                # print("preprocess feature_attention_mask: ", feature["attention_mask"])
                # print("preprocess feature_token_type_ids: ", feature["token_type_ids"])
                # print("preprocess feature_unique_id: ", feature["unique_id"])
                # print("preprocess feature_tokens_list: ", feature["tokens"])
                # print("preprocess feature_question_text: ", feature["question_text"])
                # if f_index > 0:
                #     print("preprocess feature_context_text: ", feature["context_text"])

        return rst


    def predict(self, in_data):

        return self.model_predictor.predict(in_data)


    def postprocess(self, result):

        # print("postprocess result_input_ids: ", result["input_ids"])
        # print("postprocess result_attention_mask: ", result["attention_mask"])
        # print("postprocess result_token_type_ids: ", result["token_type_ids"])
        # print("postprocess result_unique_id: ", result["unique_id"])
        # print("postprocess result_question_text: ", result["question_text"])
        # print("postprocess result_context_text: ", result["context_text"])
        # print("postprocess result_tokens_list: ", result["tokens_list"])
        # print("postprocess result_start_logits_shape: ", np.array(result["start_logits"]).shape)
        # print("postprocess result_start_logits: ", result["start_logits"])
        # print("postprocess result_end_logits_shape: ", np.array(result["end_logits"]).shape)
        # print("postprocess result_end_logits: ", result["end_logits"])
        # print("postprocess result_predictions_shape: ", np.array(result["predictions"]).shape)
        # print("postprocess result_predictions: ", result["predictions"])

        outputs = self.generate_answers(result=result)
        return outputs
