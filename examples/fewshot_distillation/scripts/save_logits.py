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

from running_utils import get_task_list, get_task_settings, options, run_task
import os
import shutil
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

assistant_model_mapping = {
    "sst-2": "mr",
    "mr": "sst-2",
    "mrpc":"qqp",
    "qqp": "mrpc",
    "mnli": "snli",
    "snli": "mnli",
    "qnli": "rte",
    "rte": "qnli",
}

model_type = "roberta_large"
result_suffix = "save_logits"


task_list = get_task_list()

for i in options.tasks:
    if options.tasks is not None and i not in options.tasks:
        continue
    task_name = task_list[i]
    template, mapping, extra = get_task_settings(task_name)
    # LOADING CROSS DATA TO IN_DOMAIN MODEL
    args_template = [
        "--task_name={task_name}",
        "--data_dir=data/k-shot/{task_name}/{data_k}-{seed}",
        "--overwrite_output_dir",
        "--model_name_or_path={load_model}",
        "--few_shot_type=prompt",
        "--num_k={k}",
        "--output_dir=results/{task_name}/{result_suffix}/{model_type}/{k}-{seed}",
        "--seed=42",
        '--template="{template}"',
        '--mapping="{mapping}"',
        "--teacher_mode",
    ]
    if extra is not None:
        args_template.append(extra)

    load_model = "results/{task_name}/{result_suffix}/{model_type}/{k}-{seed}".format(
        task_name=assistant_model_mapping[task_name],
        result_suffix="ptkd_teacher",
        model_type=model_type,
        k=options.k,
        seed=options.seed
    )

    args = " ".join(args_template).format(
        task_name=task_name,
        data_k = options.cross_k,
        k=options.k,
        cross_model=assistant_model_mapping[task_name],
        seed=options.seed,
        load_model=load_model,
        batch_size=4,
        learning_rate=1e-05,
        result_suffix=result_suffix,
        model_type=model_type,
        template=template,
        mapping=mapping,
        temperature=5,
    )
    run_task(args)

    assistant_model_path = "results/{assitant_model}/{result_suffix}/{model_type}/{k}-{seed}".format(
        assitant_model=task_name,
        result_suffix="ptkd_teacher",
        model_type=model_type,
        k=options.cross_k,
        seed=options.seed
    )
    
    logger.info(f"mapping: {task_name}->{assistant_model_mapping[task_name]}")

    output_dir = "results/{task_name}/{result_suffix}/{model_type}/{k}-{seed}".format(
        task_name=task_name,
        result_suffix=result_suffix,
        model_type=model_type,
        k=options.k,
        seed=options.seed
    )

    
    # Copy assistant cls_logits to current dirs
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copyfile(os.path.join(assistant_model_path, "cls_logits.pkl"), 
            os.path.join(output_dir, "cls_logits_ass.pkl"))
