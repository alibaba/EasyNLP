"""This script samples K examples randomly without replacement from the original data."""

# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:10 pm
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @Github  : https://github.com/alibaba/EasyTransfer, https://github.com/wjn1996

import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from utils import load_single_data, label_to_num, full_k_to_num

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style (tsv)
            dataset = {}
            dirname = os.path.join(data_dir, task)
            if task == "MNLI":
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
    if task in ["CoLA"]:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown GLUE task.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    parser.add_argument("--scene", type=str, default='few-shot', choices=['few-shot', 'full'],
                        help="The scene of data, if choose few-shot, please give k, otherwise please ignore the k")
    parser.add_argument("--task", type=str, nargs="+",
                        default=['SST-2', 'mr', 'cr', 'MNLI', 'SNLI', 'MRPC', 'QQP'],
                        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+", 
        default=[42, 21, 12],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="./data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot-single', choices=['k-shot-single'], help="k-shot-single")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    if args.scene == 'few-shot':
        print("K =", args.k)
    else:
        print("Full")
        
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)
            k = args.k
            if args.scene == 'full':
                k = full_k_to_num[task]
            # Shuffle the training set
            print("| Task = %s" % (task))
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style 
                train_header, train_lines = split_header(task, dataset["train"])
                np.random.shuffle(train_lines)
            else:
                # Other datasets 
                train_lines = dataset['train'].values.tolist()
                np.random.shuffle(train_lines)
            # print(train_lines[:5])
            # Set up dir
            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            # Write test splits 原封不动地将测试集保存下，不作为group内数据
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                new_test = []
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace('dev', 'test')
                    test_header, test_lines = split_header(task, lines)
                    np.random.shuffle(test_lines)
                    lal = load_single_data(task, test_lines)
                    # with open(os.path.join(setting_dir, f"{split}.tsv"), "w") as f:
                    for key, value in lal.items():
                        new_test += value
                    # 将验证集作为测试集，并保存
                new_test = DataFrame(new_test)
                new_test.to_csv(os.path.join(setting_dir, 'test.csv'), sep='\t', header=False, index=False)
            else:
                # Other datasets
                # Use the original test sets
                new_test = []
                test_lines = dataset['test'].values.tolist()
                np.random.shuffle(test_lines)
                lal = load_single_data(task, test_lines)
                # with open(os.path.join(setting_dir, f"{split}.tsv"), "w") as f:
                for key, value in lal.items():
                    new_test += value
                new_test = DataFrame(new_test)
                new_test.to_csv(os.path.join(setting_dir, 'test.csv'), sep='\t', header=False, index=False)
            
            # Get label list for balanced sampling
            label_list_ = load_single_data(task, train_lines)
            label_list_train_top_k = dict()
            label_list_dev_top_k = dict()
            # 对当前Task的每一类数据进行shuffle，并取前K个，作为采样的数据
            num_label = label_to_num[task]  # 当前数据集的标签数量
            for label in label_list_.keys():
                n = len(label_list_[label]) * 0.8
                # 如果K超过了原始训练集数量的0.8倍，则取前80%
                if k * num_label > n:
                    k_ = int(n)
                else:
                    k_ = k
                np.random.shuffle(label_list_[label])
                label_list_train_top_k[label] = label_list_[label][:k_]
                dev_rate = 11 if '10x' in args.mode else 2
                label_list_dev_top_k[label] = label_list_[label][k_:k_ * dev_rate]

            label_train_list = label_list_train_top_k
            label_dev_list = label_list_dev_top_k

            # 对每个类内的样本进行shuffle
            for label in label_train_list.keys():
                np.random.shuffle(label_train_list[label])
                np.random.shuffle(label_dev_list[label])
            # 取group内的所有训练集作为小样本数据集，对每一组分别处理
            new_train = []
            for label in label_train_list:
                for line in label_train_list[label]:
                    new_train.append(line)
            new_train = DataFrame(new_train)
            new_train.to_csv(os.path.join(setting_dir, 'train.csv'), sep='\t', header=False, index=False)

            new_dev = []
            for label in label_dev_list:
                dev_rate = 11 if '10x' in args.mode else 2
                for line in label_dev_list[label]:
                    new_dev.append(line)
            new_dev = DataFrame(new_dev)
            new_dev.to_csv(os.path.join(setting_dir, 'dev.csv'), sep='\t', header=False, index=False)


if __name__ == "__main__":
    main()
