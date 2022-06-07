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

teacher_model = "pretrained/roberta-large"
model_type = "roberta_large"
result_suffix = "ptkd_teacher"

args_template = [
    "--task_name={task_name}",
    "--data_dir=data/k-shot/{task_name}/{k}-{seed}",
    "--overwrite_output_dir",
    "--do_train",
    "--do_eval",
    "--do_predict",
    "--model_name_or_path={load_model}",
    "--few_shot_type=prompt",
    "--num_k={k}",
    "--eval_steps={eval_steps}",
    "--max_steps={max_steps}",
    "--per_device_train_batch_size={batch_size}",
    "--per_device_eval_batch_size=4",
    "--learning_rate={learning_rate}",
    "--output_dir=results/{task_name}/{result_suffix}/{model_type}/{k}-{seed}",
    "--seed=42",
    '--template="{template}"',
    '--mapping="{mapping}"',
    "--num_sample=1",
    "--teacher_mode",
    "--alpha={alpha}",
    "--beta={beta}",
]

task_list = get_task_list()

for i in options.tasks:
    if options.tasks is not None and i not in options.tasks:
        continue
    task_name = task_list[i]
    template, mapping, extra = get_task_settings(task_name)
    if extra is not None:
        args_template.append(extra)

    max_steps = 2 * options.k // 8 * 30
    eval_steps = max_steps // 3

    args = " ".join(args_template).format(
        task_name=task_name,
        k=options.k,
        eval_steps=eval_steps,
        max_steps=max_steps,
        seed=options.seed,
        load_model=teacher_model,
        batch_size=4,
        learning_rate=1e-05,
        result_suffix=result_suffix,
        model_type=model_type,
        template=template,
        mapping=mapping,
        alpha=0,
        beta=0,
    )

    run_task(args)
