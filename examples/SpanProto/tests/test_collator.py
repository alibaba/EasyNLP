# -*- coding: utf-8 -*-
# @Time    : 2021/11/28 5:54 pm.
# @Author  : JianingWang
# @File    : test_collator.py
import unittest
from data.data_collator import DataCollatorForLMWithoutNumber
from transformers import AutoTokenizer


class TestCollator(unittest.TestCase):
    def test_collator_without_number(self):
        tokenizer = AutoTokenizer.from_pretrained('/Users/JianingWang/models/sm_medbert_small')
        text = '这是用1来2测3试4数5字6不7会8被9用0来做防1止2数3字5无6法7被8预9测0不能收敛的情况'
        collator = DataCollatorForLMWithoutNumber(tokenizer=tokenizer, mlm_probability=0.25)
        tokenizer_out = tokenizer(text, return_special_tokens_mask=True)
        for _ in range(0, 10):
            collator_out = collator([tokenizer_out], return_tensors='pt')
            for i, j in enumerate(collator_out['input_ids'][0].tolist()):
                if j == 103:
                    print(text[i - 1])
                    assert text[i - 1] not in [str(i) for i in range(0, 10)]
            print(' ')
