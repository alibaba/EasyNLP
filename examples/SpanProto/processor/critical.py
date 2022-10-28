# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 10:27 pm.
# @Author  : JianingWang
# @File    : critical.py
import os

from transformers import DataCollatorWithPadding

from processor.ProcessorBase import CLSProcessor


class CriticalProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args)
        self.labels = ['0', '1']
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}
        self.train_file = os.path.join(data_args.data_dir, 'train.txt')
        self.dev_file = os.path.join(data_args.data_dir, 'dev.txt')

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorWithPadding(self.tokenizer, padding='longest', pad_to_multiple_of=64 if pad_to_multiple_of_8 else None)


    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self._create_examples(self._read_text(self.train_file), 'train')
            self.train_examples = examples
        elif set_type == 'dev':
            examples = self._create_examples(self._read_text(self.dev_file), 'dev')
            self.dev_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            text_a, label = line.strip().split('\t')
            assert label in self.labels
            examples.append({'text_a': text_a, 'label': label})
        return examples
