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

# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:10 pm
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @Github  : https://github.com/alibaba/EasyTransfer, https://github.com/wjn1996

"""
This file contains the logic for loading data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

import log
from pet import task_helpers
from pet.utils import InputExample
from transformers import DataProcessor as TransDataProcessor


logger = log.get_logger('root')

def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading train/dev32/dev/test/unlabeled examples for a given task.
    """

    def __init__(self, task_name):
        self.task_name = task_name

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_dev32_examples(self, data_dir) -> List[InputExample]:
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    # @abstractmethod
    # def get_labels(self) -> List[str]:
    #     """Get the list of labels for this data set."""
    #     pass


class RteProcessor(DataProcessor):
    """Processor for the RTE data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, path: str, set_type: str, hypothesis_name: str = "hypothesis",
                         premise_name: str = "premise") -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = line_idx
                label = example_json.get('label')
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json[premise_name]
                text_b = example_json[hypothesis_name]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class CbProcessor(RteProcessor):
    """Processor for the CB data set."""

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]


class WicProcessor(DataProcessor):
    """Processor for the WiC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["F", "T"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    idx = int(idx)
                label = "T" if example_json.get('label') else "F"
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['sentence1']
                text_b = example_json['sentence2']
                meta = {'word': example_json['word']}
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx, meta=meta)
                examples.append(example)
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['text']
                meta = {
                    'span1_text': example_json['target']['span1_text'],
                    'span2_text': example_json['target']['span2_text'],
                    'span1_index': example_json['target']['span1_index'],
                    'span2_index': example_json['target']['span2_index']
                }

                # the indices in the dataset are wrong for some examples, so we manually fix them
                span1_index, span1_text = meta['span1_index'], meta['span1_text']
                span2_index, span2_text = meta['span2_index'], meta['span2_text']
                words_a = text_a.split()
                words_a_lower = text_a.lower().split()
                words_span1_text = span1_text.lower().split()
                span1_len = len(words_span1_text)

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    for offset in [-1, +1]:
                        if words_a_lower[span1_index + offset:span1_index + span1_len + offset] == words_span1_text:
                            span1_index += offset

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    logger.warning(f"Got '{words_a_lower[span1_index:span1_index + span1_len]}' but expected "
                                   f"'{words_span1_text}' at index {span1_index} for '{words_a}'")

                if words_a[span2_index] != span2_text:
                    for offset in [-1, +1]:
                        if words_a[span2_index + offset] == span2_text:
                            span2_index += offset

                    if words_a[span2_index] != span2_text and words_a[span2_index].startswith(span2_text):
                        words_a = words_a[:span2_index] \
                                  + [words_a[span2_index][:len(span2_text)], words_a[span2_index][len(span2_text):]] \
                                  + words_a[span2_index + 1:]

                assert words_a[span2_index] == span2_text, \
                    f"Got '{words_a[span2_index]}' but expected '{span2_text}' at index {span2_index} for '{words_a}'"

                text_a = ' '.join(words_a)
                meta['span1_index'], meta['span2_index'] = span1_index, span2_index

                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                if set_type == 'train' and label != 'True':
                    continue
                examples.append(example)

        return examples


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["False", "True"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = str(example_json['label']) if 'label' in example_json else None
                idx = example_json['idx']
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['premise']
                meta = {
                    'choice1': example_json['choice1'],
                    'choice2': example_json['choice2'],
                    'question': example_json['question']
                }
                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                examples.append(example)

        if set_type == 'train' or set_type == 'unlabeled':
            mirror_examples = []
            for ex in examples:
                label = "1" if ex.label == "0" else "0"
                meta = {
                    'choice1': ex.meta['choice2'],
                    'choice2': ex.meta['choice1'],
                    'question': ex.meta['question']
                }
                mirror_example = InputExample(guid=ex.guid + 'm', text_a=ex.text_a, label=label, meta=meta)
                mirror_examples.append(mirror_example)
            examples += mirror_examples
            logger.info(f"Added {len(mirror_examples)} mirror examples, total size is {len(examples)}...")
        return examples


class MultiRcProcessor(DataProcessor):
    """Processor for the MultiRC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json['idx']
                text = example_json['passage']['text']
                questions = example_json['passage']['questions']
                for question_json in questions:
                    question = question_json["question"]
                    question_idx = question_json['idx']
                    answers = question_json["answers"]
                    for answer_json in answers:
                        label = str(answer_json["label"]) if 'label' in answer_json else None
                        answer_idx = answer_json["idx"]
                        guid = f'{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}'
                        meta = {
                            'passage_idx': passage_idx,
                            'question_idx': question_idx,
                            'answer_idx': answer_idx,
                            'answer': answer_json["text"]
                        }
                        idx = [passage_idx, question_idx, answer_idx]
                        example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev32.jsonl"), "dev32")

    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path, set_type, seed=42, max_train_candidates_per_question: int = 10) -> List[InputExample]:
        examples = []

        entity_shuffler = random.Random(seed)

        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)

                idx = example_json['idx']
                text = example_json['passage']['text']
                entities = set()

                for entity_json in example_json['passage']['entities']:
                    start = entity_json['start']
                    end = entity_json['end']
                    entity = text[start:end + 1]
                    entities.add(entity)

                entities = list(entities)

                text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations
                questions = example_json['qas']

                for question_json in questions:
                    question = question_json['query']
                    question_idx = question_json['idx']
                    answers = set()

                    for answer_json in question_json.get('answers', []):
                        answer = answer_json['text']
                        answers.add(answer)

                    answers = list(answers)

                    if set_type == 'train':
                        # create a single example per *correct* answer
                        for answer_idx, answer in enumerate(answers):
                            candidates = [ent for ent in entities if ent not in answers]
                            if len(candidates) > max_train_candidates_per_question - 1:
                                entity_shuffler.shuffle(candidates)
                                candidates = candidates[:max_train_candidates_per_question - 1]

                            guid = f'{set_type}-p{idx}-q{question_idx}-a{answer_idx}'
                            meta = {
                                'passage_idx': idx,
                                'question_idx': question_idx,
                                'candidates': [answer] + candidates,
                                'answers': [answer]
                            }
                            ex_idx = [idx, question_idx, answer_idx]
                            example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta,
                                                   idx=ex_idx)
                            examples.append(example)

                    else:
                        # create just one example with *all* correct answers and *all* answer candidates
                        guid = f'{set_type}-p{idx}-q{question_idx}'
                        meta = {
                            'passage_idx': idx,
                            'question_idx': question_idx,
                            'candidates': entities,
                            'answers': answers
                        }
                        example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
                    f"distribution {list(label_distribution.items())}")
        return examples




### add by wjn 文本分类任务的处理（取自LM-BFF）
import pandas as pd

class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        # self.task_name = task_name
        """See base class."""
        print('task_name = ', self.task_name)
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        elif self.task_name == 'g1':
            return list(range(2))
        else:
            raise Exception("task_name not supported.")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa', 'g1']:
                examples.append(InputExample(guid=guid, text_a=line[0], text_b=None, task=line[1], label=line[2]))
            else:
                raise Exception("Task_name not supported.")

        return examples

class G1Processor(DataProcessor):
    """
    Data processor for group1 datasets.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_each_task_examples(self, examples: List[InputExample]):
        task_to_example = dict()  # {task_name: [.., ..], ..}
        for example in examples:
            if example.task not in task_to_example.keys():
                task_to_example[example.task] = []
            task_to_example[example.task].append(example)
        return task_to_example

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=None, task=line[1], label=line[2]))
        return examples



### add by wjn SST-2任务的处理
class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=None, task=line[1], label=line[2]))
        return examples

class MrProcessor(DataProcessor):
    """
    Data processor for MR datasets.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=None, task=line[1], label=line[2]))
        return examples

class CrProcessor(DataProcessor):
    """
    Data processor for CR datasets.
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=None, task=line[1], label=line[2]))
        return examples


### add by wjn
class G2Processor(DataProcessor):
    """Processor for the Group2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_each_task_examples(self, examples: List[InputExample]):
        task_to_example = dict() # {task_name: [.., ..], ..}
        for example in examples:
            if example.task not in task_to_example.keys():
                task_to_example[example.task] = []
            task_to_example[example.task].append(example)
        return task_to_example

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=line[1], task=line[2], label=line[3]))
        return examples


### add by wjn
class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=line[1], task=line[2], label=line[3]))
        return examples

### add by wjn
class SnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=line[1], task=line[2], label=line[3]))
        return examples

## add by wjn
class G3Processor(DataProcessor):
    """Processor for the Group3 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_each_task_examples(self, examples: List[InputExample]):
        task_to_example = dict()  # {task_name: [.., ..], ..}
        for example in examples:
            if example.task not in task_to_example.keys():
                task_to_example[example.task] = []
            task_to_example[example.task].append(example)
        return task_to_example

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 0
        q2_index = 1
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[3]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, task=line[2], label=label))
        return examples

### add by wjn
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text_a=line[0], text_b=line[1], task=line[2], label=line[3]))
        return examples


### add by wjn
class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(),
                                     "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_dev32_examples(self, data_dir):
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), sep='\t', header=None, keep_default_na=False).values.tolist(), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 0
        q2_index = 1
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = line[3]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, task=line[2], label=label))
        return examples


PROCESSORS = {
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor,
    ### add by wjn 下面是用于完成一些其他Task的实验 ###
    "g1": G1Processor, # cross task group 1
    "sst-2": Sst2Processor, # SST-2
    "mr": MrProcessor, # MR
    "cr": CrProcessor, # CR
    "g2": G2Processor,
    "mnli": MnliProcessor,
    "snli": SnliProcessor,
    "g3": G3Processor,
    "mrpc": MrpcProcessor,
    "qqp": QqpProcessor,
}  # type: Dict[str,Callable[[],DataProcessor]]


TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    # "record": task_helpers.RecordTaskHelper,
}

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"],
    "record": ["acc", "f1"]
}

DEFAULT_METRICS = ["acc"]


TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"
DEV32_SET = "dev32"
SPE_TRAIN_SET = "spe_train" # add by wjn 在cross-task混合数据group中，获得其每一个task对应的训练样本
SPE_DEV_SET = "spe_dev" # add by wjn 在cross-task混合数据group中，获得其每一个task对应的验证样本

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET, DEV32_SET]


def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42,
                  examples: List[InputExample] = None) -> List[InputExample]:
    """Load examples for a given task."""

    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task](task)

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == DEV32_SET: ### TODO
        examples = processor.get_dev32_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    elif set_type == SPE_TRAIN_SET:
        return processor.get_each_task_examples(examples)
    elif set_type == SPE_DEV_SET:
        return processor.get_each_task_examples(examples)
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")

    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples
