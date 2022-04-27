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

import argparse


def add_basic_argument():
    parser = argparse.ArgumentParser()

    basic_group = parser.add_argument_group('basic-args')
    basic_group.add_argument("--mode",
                             default="train",
                             type=str,
                             required=True,
                             choices=["train", "evaluate", "predict"],
                             help="The mode of easytexminer")
    basic_group.add_argument("--tables",
                             default=None,
                             type=str,
                             required=True,
                             help="The input table, "
                                  "`train_file`,`valid_file` if mode=`train`;"
                                  "`valid_file` if mode=`evaluate`")
    basic_group.add_argument("--log_file",
                             default=None,
                             type=str,
                             help="logging files saved path")
    basic_group.add_argument("--logging_steps",
                             default=100,
                             type=int,
                             help="logging steps while training")
    basic_group.add_argument('--seed',
                             type=int,
                             default=42,
                             help="random seed for initialization")
    basic_group.add_argument('--buckets',
                             type=str,
                             default=None,
                             help="Oss buckets")
    basic_group.add_argument("--model_name",
                               default="text_classify_bert",
                               type=str,
                               choices=["text_classify_bert",
                                        "text_classify_bert_emtl",
                                        "text_match_bert",
                                        "text_match_bert_two_tower",
                                        "language_modeling_bert",
                                        "sequence_labeling_bert",
                                        "text_classify_tinybert",
                                        "text_match_tinybert",
                                        "sequence_labeling_tinybert"],
                               help="Type of the model")
    basic_group.add_argument("--checkpoint_dir", "--checkpoint_path",
                               default=None,
                               type=str,
                               help="The model checkpoint dir.")
    basic_group.add_argument("--pretrain_model_name_or_path",
                               default=None,
                               type=str,
                               required=False,
                               help="The pretrain model name or path.")

    preprocessor_group = parser.add_argument_group('preprocessor-args')
    preprocessor_group.add_argument("--sequence_length",
                                    default=128,
                                    type=int,
                                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                                         "Sequences longer than this will be truncated, and sequences shorter \n"
                                         "than this will be padded.")
    preprocessor_group.add_argument("--do_lower_case",
                                    action='store_true',
                                    help="Set this flag if you are using an uncased model.")

    train_group = parser.add_argument_group('train-args')
    train_group.add_argument("--train_batch_size",
                             default=32,
                             type=int,
                             help="Total batch size for training.")
    train_group.add_argument("--eval_batch_size",
                             default=32,
                             type=int,
                             help="Total batch size for evaluation.")
    train_group.add_argument("--epoch_num",
                             default=3.0,
                             type=float,
                             help="Total number of training epochs to perform.")
    train_group.add_argument('--save_checkpoint_steps',
                             type=int,
                             default=None)
    train_group.add_argument('--save_all_checkpoints',
                             action='store_true',
                             help="Whether to save all checkpoints per eval steps.")
    train_group.add_argument("--learning_rate",
                             default=5e-5,
                             type=float,
                             help="The initial learning rate for Adam.")
    train_group.add_argument('--weight_decay', '--wd',
                             default=1e-4,
                             type=float,
                             metavar='W',
                             help='weight decay')
    train_group.add_argument('--max_grad_norm', '--mn',
                             default=1.0,
                             type=float,
                             help='Max grad norm')
    train_group.add_argument("--warmup_proportion",
                             default=0.1,
                             type=float,
                             help="Proportion of training to perform linear learning rate warmup for. "
                                  "E.g., 0.1 = 10%% of training.")
    train_group.add_argument('--gradient_accumulation_steps',
                             type=int,
                             default=1,
                             help="Number of updates steps to accumulate before performing a backward/update pass.")
    train_group.add_argument('--resume_from_checkpoint',
                             type=str,
                             default=None,
                             help="Resume training process from checkpoint")
    train_group.add_argument('--export_tf_checkpoint_type',
                             type=str,
                             default="easytransfer",
                             choices=["easytransfer", "google"],
                             help="Which type of checkpoint you want to export")

    dist_group = parser.add_argument_group('distribution-args')
    dist_group.add_argument('--local_rank', default=-1, type=int, help='Node rank for distributed training')
    dist_group.add_argument('--worker_count', default=1, type=int, help='Count of workers/servers')
    dist_group.add_argument('--gpu_count', default=-1, type=int, help='Count of GPUs in each worker')
    dist_group.add_argument('--cpu_count', default=-1, type=int, help='Count of CPUs in each worker')
    dist_group.add_argument('--master_port', default=23456, type=int, help='Port of master node')

    app_group = parser.add_argument_group('app-args')
    app_group.add_argument("--app_name",
                             default="",
                             type=str,
                             choices=["classification",
                                      "distillation"],
                             help="Name of the App")
    app_group.add_argument("--input_schema", type=str, default=None,
                           help='Only for csv data, the schema of input table')
    app_group.add_argument("--first_sequence", type=str, default=None,
                           help='Which column is the first sequence mapping to')
    app_group.add_argument("--second_sequence", type=str, default=None,
                           help='Which column is the second sequence mapping to')
    app_group.add_argument("--label_name", type=str, default=None,
                           help='Which column is the label mapping to')
    app_group.add_argument("--label_enumerate_values", type=str, default=None,
                           help='Which column is the label mapping to')

    return parser


def add_distill_argument(parser):
    distill_group = parser.add_argument_group('distillation-args')
    distill_group.add_argument("--teacher_type",
                               default="text_classify_bert",
                               type=str,
                               choices=["text_classify_bert",
                                        "text_classify_bert_emtl",
                                        "text_match_bert",
                                        "text_match_bert_two_tower",
                                        "language_modeling_bert",
                                        "sequence_labeling_bert",
                                        "text_classify_tinybert",
                                        "text_match_tinybert",
                                        "sequence_labeling_tinybert"],
                               help="Type of the teacher model")
    distill_group.add_argument("--teacher_dir",
                               default=None,
                               type=str,
                               help="The teacher model dir. Can be None if mode=`evaluate`")
    distill_group.add_argument("--not_use_teacher_vocab",
                               default=False,
                               action='store_true',
                               help="Whether to copy teacher's vocab")
    distill_group.add_argument("--student_init_strategy",
                               default=None,
                               type=str,
                               required=False,
                               choices=["skip", "first", "last"],
                               help="The pretrain student model dir.")
    distill_group.add_argument('--temperature',
                               type=float,
                               default=1.)
    distill_group.add_argument("--distill_method", type=str, default=None,
                               help='Which distillation method you want to use')
    distill_group.add_argument("--student_config", type=str, default=None,
                               help='Student config setting')

    return parser


def add_mtl_argument(parser):
    mtl_group = parser.add_argument_group('multitask-args')
    # Arguments for multi-task text classification
    mtl_group.add_argument("--task_column_name", type=str, default=None,
                           help='Which column is the task_key mapping to, only for multi-task learning')
    mtl_group.add_argument("--freeze_encoder", default=False, action="store_true",
                           help='Freeze the encoder if it is true.')
    mtl_group.add_argument("--max_task_num", type=int, default=None,
                           help='Maximum number of tasks.')
    mtl_group.add_argument("--max_label_num", type=int, default=None,
                           help='Maximum number of label for each task.')
    return parser
