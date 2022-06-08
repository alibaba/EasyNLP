# coding=utf-8
# Copyright (c) 2021 Princeton Natural Language Processing.
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

"""This script samples K examples randomly without replacement from the original data."""

import argparse
import os
from secrets import choice
import numpy as np
import pandas as pd
from pandas import DataFrame

def get_label(task, line):
    if task in ["mnli", "mrpc", "qnli", "qqp", "rte", "snli", "sst-2", "sts-b", "wnli", "cola"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'cola':
            return line[1]
        elif task == 'mnli':
            return line[-1]
        elif task == 'mrpc':
            return line[0]
        elif task == 'qnli':
            return line[-1]
        elif task == 'qqp':
            return line[-1]
        elif task == 'rte':
            return line[-1]
        elif task == 'snli':
            return line[-1]
        elif task == 'sst-2':
            return line[-1]
        elif task == 'sts-b':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'wnli':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["mnli", "mrpc", "qnli", "qqp", "rte", "snli", "sst-2", "sts-b", "wnli", "cola"]:
            # GLUE style (tsv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "mnli":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            # Other datasets (csv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets

def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task in ["cola"]:
        return [], lines
    elif task in ["mnli", "mrpc", "qnli", "qqp", "rte", "snli", "sst-2", "sts-b", "wnli"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    parser.add_argument("--task", type=str, nargs="+", 
        default=['sst-2', 'sst-5', 'mr', 'cr', 'mpqa', 'subj', 'trec', 'cola', 'mrpc', 'qqp', 'sts-b', 'mnli', 'snli', 'qnli', 'rte'],
        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+", 
        default=[100, 13, 21, 42, 87],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'], help="k-shot or k-shot-10x (10x dev set)")
    parser.add_argument("--cross", type=str, default='k-shot', choices=['k-shot','k-shot-2x','k-shot-3x','k-shot-4x','k-shot-5x',
        'k-shot-6x', 'k-shot-7x','k-shot-8x','k-shot-9x', 'k-shot-10x'], help="k-shot or k-shot-10x (10x train set)")
    parser.add_argument("--full", action="store_true", help="full_set")


    args = parser.parse_args()
    if args.full:
        args.output_dir = os.path.join(args.output_dir, "full")
        args.seed = [13]
    else:
        args.output_dir = os.path.join(args.output_dir, args.cross)

    k = args.k
    print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))
            if task in ["mnli", "mrpc", "qnli", "qqp", "rte", "snli", "sst-2", "sts-b", "wnli", "cola"]:
                # GLUE style 
                train_header, train_lines = split_header(task, dataset["train"])
                np.random.shuffle(train_lines)
            else:
                # Other datasets 
                train_lines = dataset['train'].values.tolist()
                np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            if args.full:
                setting_dir = task_dir
            else:
                setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            
            os.makedirs(setting_dir, exist_ok=True)

            # Write test splits
            if task in ["mnli", "mrpc", "qnli", "qqp", "rte", "snli", "sst-2", "sts-b", "wnli", "cola"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace('dev', 'test')
                    with open(os.path.join(setting_dir, f"{split}.tsv"), "w") as f:
                        for line in lines:
                            f.write(line)
            else:
                # Other datasets
                # Use the original test sets
                dataset['test'].to_csv(os.path.join(setting_dir, 'test.csv'), header=False, index=False)  
            
            # Get label list for balanced sampling
            label_list = {}
            for line in train_lines:
                label = get_label(task, line)
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)
            
            if task in ["mnli", "mrpc", "qnli", "qqp", "rte", "snli", "sst-2", "sts-b", "wnli", "cola"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        train_rate = 11 if '10x' in args.cross else 2
                        if args.full:
                            for line in label_list[label][:]:
                                f.write(line)
                        else:
                            for line in label_list[label][k:k*train_rate]:
                                f.write(line)
                name = "dev.tsv"
                if task == 'mnli':
                    name = "dev_matched.tsv"
                with open(os.path.join(setting_dir, name), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        dev_rate = 11 if '10x' in args.mode else 2
                        for line in label_list[label][k:k*dev_rate]:
                            f.write(line)
            else:
                new_train = []
                for label in label_list:
                    train_rate = 11 if '10x' in args.cross else 2
                    if args.full:
                        for line in label_list[label][:]:
                            new_train.append(line)
                    else:
                        for line in label_list[label][k:k*train_rate]:
                            new_train.append(line)
                new_train = DataFrame(new_train)
                new_train.to_csv(os.path.join(setting_dir, 'train.csv'), header=False, index=False)
            
                new_dev = []
                for label in label_list:
                    dev_rate = 11 if '10x' in args.mode else 2
                    for line in label_list[label][k:k*dev_rate]:
                        new_dev.append(line)
                new_dev = DataFrame(new_dev)
                new_dev.to_csv(os.path.join(setting_dir, 'dev.csv'), header=False, index=False)


if __name__ == "__main__":
    main()
