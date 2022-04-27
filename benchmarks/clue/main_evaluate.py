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

import sys
import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from easynlp.appzoo.api import get_application_evaluator
from easynlp.core.trainer import Trainer
from easynlp.utils import initialize_easynlp, get_args, get_pretrain_model_path
from easynlp.utils.global_vars import parse_user_defined_parameters
from benchmarks.clue.application import CLUEApp
from benchmarks.clue.utils import load_dataset
from benchmarks.clue.preprocess import tasks2processor


if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    print('args.learning_rate=', args.learning_rate)

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == "train" or not args.checkpoint_dir:
        args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir
    args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)

    print('pretrained_model_name_or_path', args.pretrained_model_name_or_path)

    clue_name = user_defined_parameters.get("clue_name", "clue")
    task_name = user_defined_parameters.get("task_name", "csl")
    num_labels = int(user_defined_parameters.get("num_labels", 2))
    assert task_name in tasks2processor.keys()

    preprocessor = tasks2processor[task_name](
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        max_seq_length=args.sequence_length,
    )

    dataset = load_dataset(
        clue_name=clue_name,
        task_name=task_name,
        preprocessor=preprocessor,
    )

    model = CLUEApp(
        args=args,
        task_name=task_name,
        app_name=args.app_name,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        user_defined_parameters=user_defined_parameters,
        is_training=False,
        num_labels=len(preprocessor.get_labels()),
    )

    model.to(torch.cuda.current_device())

    evaluator = get_application_evaluator(
        app_name=args.app_name,
        valid_dataset=dataset["dev"],
        user_defined_parameters=user_defined_parameters,
        eval_batch_size=args.micro_batch_size
    )

    evaluator.evaluate(model=model)


