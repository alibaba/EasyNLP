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
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.

在multi-task language learner + task generalization，
（1）首先加载group1内的cross-task数据（一共3个task，记做a,b,c），任意挑选两个task（a, b）用于训练meta-learner
（2）其次在group1内的剩余的任务上（c）单独进行微调，其中prompt encoder的参数需要借助task representation相似度进行初始化
"""

import argparse
import os
from typing import Tuple
import torch

from data_utils.task_processors import PROCESSORS, load_examples, DEV32_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from data_utils.utils import data_to_name, groups
from pet.utils import eq_div
from pet.wrapper import MODEL_CLASSES

from pet.config import TrainConfig, EvalConfig, WrapperConfig
from pet.modeling import train_pet, train_pet_cross, train_adaptation_cross, train_generalization_cross

import log
logger = log.get_logger('root')


def load_pet_configs(args) -> Tuple[WrapperConfig, TrainConfig, EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type,
                              model_name_or_path=args.model_name_or_path,
                              data_dir=args.data_dir, # add by wjn
                              task_type=args.task_type, # add by wjn
                              task_name=args.task_name,
                              k=args.k, # add by wjn 每个标签对应的样本数，取决于LM-BFF模型中生成数据的K值
                              label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length,
                              cache_dir=args.cache_dir,
                              output_dir=args.output_dir,
                              embed_size=args.embed_size,
                              prompt_encoder_type=args.prompt_encoder_type,
                              eval_every_step=args.eval_every_step,
                              scene=args.scene)

    train_cfg = TrainConfig(device=args.device,
                            per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                            n_gpu=args.n_gpu,
                            num_train_epochs=args.pet_num_train_epochs,
                            max_steps=args.pet_max_meta_steps,
                            gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                            weight_decay=args.weight_decay,
                            learning_rate=args.learning_rate,
                            adam_epsilon=args.adam_epsilon,
                            warmup_steps=args.warmup_steps,
                            max_grad_norm=args.max_grad_norm,
                            alpha=args.alpha)

    eval_cfg = EvalConfig(device=args.device,
                          n_gpu=args.n_gpu,
                          metrics=args.metrics,
                          max_steps=args.pet_max_adaptation_steps,
                          per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def main():
    parser = argparse.ArgumentParser(description="Command line interface for P-Tuning.")

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default="albert", type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_type", default='cross_task', type=str, required=False, choices=['single_task', 'cross_task'],
                        help="The type of the task to train/evaluate on") # add by wjn
    parser.add_argument("--task_name", default=['g1'], type=str, required=True, choices=['g1'],
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--k", default=16, type=int, required=False,
                        help="The number of examples of each label") # add by wjn
    parser.add_argument("--scene", default="few-shot", type=str, required=True, choices=['few-shot', 'full'],
                        help="The scene of data, if choose few-shot, please give k, otherwise please ignore the k")  # add by wjn
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--pattern_ids", default=[1], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--cross_prompt", action='store_true',
                        help="If true, when task_type is cross-task, each task in one group has different specific PVPs,"
                             "If false, all the task in one group share the same PVPs")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--pet_repetitions", default=3, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_meta_steps", default=-1, type=int,
                        help="If > 0: set total number of multi-task meta-learning training steps to perform in PET. Override num_train_epochs.")
    parser.add_argument("--pet_max_adaptation_steps", default=-1, type=int,
                        help="If > 0: set total number of task-specific adaptation training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--eval_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--dev32_examples", default=-1, type=int,
                        help="The total number of dev32 examples to use, where -1 equals all examples.")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")
    parser.add_argument("--embed_size", default=128, type=int, help="albert: 128, roberta-large:1024, roberta-base:768")
    parser.add_argument('--prompt_encoder_type', type=str, default="lstm", choices=['lstm', 'mlp'])
    parser.add_argument("--eval_every_step", default=20, type=int, help="")


    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
    #         and args.do_train and not args.overwrite_output_dir:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name](args.task_name)
    # if args.task_name in ['g1', 'mr', 'cr']:
    #     args.label_list = processor.get_labels(args.task_name)
    # else:
    args.label_list = processor.get_labels()


    train_ex_per_label, eval_ex_per_label, dev32_ex_per_label = None, None, None
    train_ex, eval_ex, dev32_ex = args.train_examples, args.eval_examples, args.dev32_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        eval_ex_per_label = eq_div(args.eval_examples, len(args.label_list)) if args.eval_examples != -1 else -1
        dev32_ex_per_label = eq_div(args.dev32_examples, len(args.label_list)) if args.dev32_examples != -1 else -1
        train_ex, eval_ex, dev32_ex = None, None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    # task adaptation 只支持cross-task
    assert args.task_type == 'cross_task'

    # 先加在group1内cross-task的所有数据
    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=-1, num_examples_per_label=None)

    dev_data = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=-1, num_examples_per_label=None)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)  # cross-task group 的 metrics

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)

    logger.info("************Training Example:**************")
    logger.info("text_a={}".format(train_data[0].text_a))
    logger.info("text_b={}".format(train_data[0].text_b))
    logger.info("task={}".format(train_data[0].task))
    logger.info("label={}".format(train_data[0].label))
    logger.info("**********************************")

    for unseen_task in groups['g1']:

        seen_task = set()

        ## 将group1内的每个task分离出来
        seen_task_train_data, seen_task_dev_data = [], []
        unseen_task_train_data, unseen_task_dev_data = [], []
        for i in train_data:
            if i.task != data_to_name[unseen_task]:
                seen_task_train_data.append(i)
                seen_task_dev_data.append(i)
                seen_task.add(i.task)
            else:
                unseen_task_train_data.append(i)
                unseen_task_dev_data.append(i)

        logger.info("======== Meta-learning on task {}, Task generalization on unseen tasks: {} ========".format(','.join(seen_task), unseen_task))

        # 执行multi-task meta-learning
        train_generalization_cross(
                  seen_task_train_data=seen_task_train_data,  # 相当于训练集
                  seen_task_dev_data=seen_task_dev_data,  # 相当于验证集
                  unseen_task_train_data=unseen_task_train_data,
                  unseen_task_dev_data=unseen_task_dev_data,
                  unseen_task=unseen_task,
                  train_config=pet_train_cfg,
                  eval_config=pet_eval_cfg,
                  model_config=pet_model_cfg,
                  pattern_ids=args.pattern_ids,
                  output_dir=args.output_dir,
                  repetitions=args.pet_repetitions,
                  do_train=args.do_train,
                  do_eval=args.do_eval,
                  seed=args.seed)

if __name__ == "__main__":
    main()
