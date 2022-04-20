"""This script samples K examples randomly without replacement from the original data."""

# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:10 pm
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @Github  : https://github.com/alibaba/EasyTransfer, https://github.com/wjn1996

'''
add by wjn
time:2021.4.19

概率采样
参考：https://www.yuque.com/minghui-gflsp/zcot1m/tgmupk
- Full场景下的Cross-Task：对于同一组内的所有Task，都包含相同的标签体系。
对于每一个类c ，计算该类下每个Task的样本的个数，根据对应个数获得该Task在类c下的采样概率
- Few场景下的Cross-Task：每个Task的每个类随机采样K个样本，作为训练集，验证集与训练集数量相同。


'''

import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from utils import groups, data_to_name, label_to_num, task_to_id, load_single_data, group_to_label, full_k_to_num


def load_datasets(data_dir, task):
    # 加载列出的所有数据集，并依次保存在dict中
    dataset = dict()
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style (tsv)
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
    else:
        # Other datasets (csv)
        dirname = os.path.join(data_dir, task)
        splits = ["train", "test"]
        for split in splits:
            filename = os.path.join(dirname, f"{split}.csv")
            dataset[split] = pd.read_csv(filename, header=None)
    return dataset

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
    parser.add_argument("--gamma", type=int, default=1,
                        help="The smoothing factor of probabilistic sampling")
    parser.add_argument("--scene", type=str, default='few-shot', choices=['few-shot', 'full'],
                        help="The scene of data, if choose few-shot, please give k, otherwise please ignore the k")
    parser.add_argument("--group", type=str, nargs="+",
        default=['g1', 'g2', 'g3'],
        help="Group names")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[42, 21, 12],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="./data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot-cross', choices=['k-shot-cross'], help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.mode)

    if args.scene == 'few-shot':
        print("Few-shot Scene with K =", args.k)
    else:
        print("Full Scene")
    # datasets = load_datasets(args.data_dir, args.task) # dict: {'task_name': {'train': xxx, 'test': xxx}, ...}
    # print(datasets['mr'])
    # 设定不同的随机种子，对于每一个随机种子，对所有的task的每一个类进行采样
    for seed in args.seed:
        print("Seed = %d" % (seed))
        # 遍历每一个group，获得group名称以及tasks
        for group in args.group:
            assert group in groups.keys()
            tasks = groups[group]
            # label_group_data 格式：{label1: {task1: [[..], ..], task2: ..}, ..}
            label_group_data = dict() # 保存当前group内，每个类对应的各个Task的样本
            assert group in group_to_label.keys()
            for label in group_to_label[group]:
                label_group_data[label] = dict()

            label_train_list, label_dev_list = dict(), dict()
            group_k = args.k
            if args.scene == 'full':
                group_k = 'full'

            # Set up group dir
            group_dir = os.path.join(args.output_dir, group)
            group_setting_dir = os.path.join(group_dir, f"{group_k}-{seed}")
            os.makedirs(group_setting_dir, exist_ok=True)

            # Set random seed
            np.random.seed(seed)
            print("| Group = %s" % (group))
            # 遍历当前组内的所有task
            for task in tasks:
                task = data_to_name[task]
                dataset = load_datasets(args.data_dir, task)  # dict: {'train': xxx, 'test': xxx}

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

                # Set up task dir
                task_dir = os.path.join(args.output_dir, task)
                task_setting_dir = os.path.join(task_dir, f"{group_k}-{seed}")
                os.makedirs(task_setting_dir, exist_ok=True)

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
                    new_test.to_csv(os.path.join(task_setting_dir, 'test.csv'), sep='\t', header=False, index=False)
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
                    new_test.to_csv(os.path.join(task_setting_dir, 'test.csv'), sep='\t', header=False, index=False)
                    # dataset['test'].to_csv(os.path.join(task_setting_dir, 'test.csv'), header=False, index=False)

                # edit by wjn
                # 获得当前group内某一个task的原始数据集 格式 {label1: [[xx], .., ], ..}
                label_list_ = load_single_data(task, train_lines)
                # 将group内所有task的按照对应的label混合起来
                for label, samples in label_list_.items():
                    if task not in label_group_data[label].keys():
                        label_group_data[label][task] = list()
                    label_group_data[label][task] += samples

            # 对group的训练集进行数据统计
            # label_sample_num: 统计整个group内，每个label对应的各个task的样本数，
            # sample_num: 统计group内所有样本个数
            # sample_num = 0
            label_sample_num = dict() # 格式：{label1: [11, 5, 12], label2: [17, 11, 14]}
            for label, task_data in label_group_data.items():
                task_num = [0] * len(tasks)
                for task_name in task_data.keys():
                    np.random.shuffle(task_data[task_name]) # 对某个label下某个task的数据进行shuffle
                    # sample_num += len(task_data[task_name])
                    task_num[task_to_id[task_name]] = len(task_data[task_name])
                label_sample_num[label] = task_num

            if args.scene == 'few-shot':
                # 对于小样本场景，group内的每个label的每个task随机采样k个样本。
                # 如果group有m个类，n个task，则一共有m*n*k个样本
                k = args.k
                for label, task_data in label_group_data.items():
                    label_train_list[label] = list()
                    label_dev_list[label] = list()
                    for task_name, data_list in task_data.items():
                        label_train_list[label] += data_list[:k] # 前k个作为训练集
                        label_dev_list[label] += data_list[k: 2*k] # k+1 - 2k作为验证集
            elif args.scene == 'full':
                # 对于全量场景，首先指定样本采样率（例如取0.8，表示80%数据作为训练集，剩余的为验证集）
                rate = 0.85
                # 对于group的每个label，根据样本总数，获得一定的label级别的采样概率
                for label, task_data in label_group_data.items():
                    label_train_list[label] = list()
                    label_dev_list[label] = list()
                    task_num = label_sample_num[label] # [x, x, x]
                    print('task_num=', task_num)
                    train_num = sum(task_num) * rate
                    # 根据公式 (log(ni) + gamma) / \sum{log(nt) + gamma}，
                    # 获得当前label对应每个task的采样概率
                    pro_samp = (np.log(task_num) + args.gamma) / sum(np.log(task_num) + args.gamma)
                    print('pro_samp=', pro_samp)
                    for task_name, data_list in task_data.items():
                        id = task_to_id[task_name]
                        k = int(train_num * pro_samp[id]) + 1
                        if k > task_num[id] * rate:
                            k = int(task_num[id] * rate)
                        print("group: {}, task: {}, label: {}, k={} ".format(group, task_name, label, k))
                        label_train_list[label] += data_list[:k]
                        label_dev_list[label] += data_list[k:]
            else:
                raise RuntimeError("Please input a validate scene in args")

            # 取group内的所有训练集作为小样本数据集，对每一组分别处理
            if group in ['g1', 'g2', 'g3']:
                new_train = []
                for label in label_train_list:
                    for line in label_train_list[label]:
                        new_train.append(line)
                new_train = DataFrame(new_train)
                new_train.to_csv(os.path.join(group_setting_dir, 'train.csv'), sep='\t', header=False, index=False)

                new_dev = []
                for label in label_dev_list:
                    for line in label_dev_list[label]:
                        new_dev.append(line)
                new_dev = DataFrame(new_dev)
                new_dev.to_csv(os.path.join(group_setting_dir, 'dev.csv'), sep='\t', header=False, index=False)

if __name__ == "__main__":
    main()
