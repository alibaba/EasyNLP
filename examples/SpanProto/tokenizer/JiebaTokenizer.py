# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 12:07 am.
# @Author  : JianingWang
# @File    : JiebaTokenizer
import jieba
from transformers import BertTokenizer


class JiebaTokenizer(BertTokenizer):
    def __init__(
            self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens
