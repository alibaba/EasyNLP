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
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The DataProcessor is responsible for loading training and test data.
This file shows an example of a DataProcessor for a new task.
"""

import csv
import os
from typing import List

from task_processors import DataProcessor, PROCESSORS
from pet.utils import InputExample


class MyTaskDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "my-task"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "dev.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["1", "2", "3", "4"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's text b or to -1 if there is no text b
    TEXT_B_COLUMN = 2

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 0

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :return: a list of train examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.TRAIN_FILE_NAME), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.DEV_FILE_NAME), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.TEST_FILE_NAME), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MyTaskDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MyTaskDataProcessor.LABELS

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                guid = "%s-%s" % (set_type, idx)
                label = row[MyTaskDataProcessor.LABEL_COLUMN]
                text_a = row[MyTaskDataProcessor.TEXT_A_COLUMN]
                text_b = row[MyTaskDataProcessor.TEXT_B_COLUMN] if MyTaskDataProcessor.TEXT_B_COLUMN >= 0 else None
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


# register the processor for this task with its name
PROCESSORS[MyTaskDataProcessor.TASK_NAME] = MyTaskDataProcessor
