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

same_task_bank = {
    "sentiment": "sst-2, mr, cr",
    "NLI_3": "mnli, snli",
    "NLI_2": "QNLI, RTE",
    "paraphrase": "MRPC, QQP",
}

# load_model = "pretrained/bert-small-uncased"
teacher_suffix = "ptkd_teacher"
teacher_type = "roberta_large"
model_type = "bert_small"
result_suffix = "cptkd_student_weights_man"

task_list = get_task_list()

for i in options.tasks:
    if options.tasks is not None and i not in options.tasks:
        continue
    task_name = task_list[i]

    max_steps = 3 * options.k // 8 * 30
    eval_steps = max_steps // 3

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
        "--save_logit_dir=results/{task_name}/{teacher_suffix}/{teacher_type}/{k}-{seed}",
        "--seed={seed}",
        '--template="{template}"',
        '--mapping="{mapping}"',
        "--num_sample=1",
        "--student_mode",
        "--with_high_prob",
        "--alpha={alpha}",
        "--temperature={temperature}",
    ]
    # TODO
    load_model = "results/{assitant_model}/{result_suffix}/{model_type}/{k}-{seed}".format(
        assitant_model=assistant_model_mapping[task_name],
        result_suffix="ptkd_student_weights_man",
        model_type="bert_small",
        k=options.k * 10,
        seed=options.seed
    )

    template, mapping, extra = get_task_settings(task_name)
    if extra is not None:
        args_template.append(extra)

    args = " ".join(args_template).format(
        task_name=task_name,
        k=options.k,
        seed=options.seed,
        load_model=load_model,
        batch_size=4,
        learning_rate=1e-05,
        max_steps=max_steps,
        eval_steps=eval_steps,
        result_suffix=result_suffix,
        teacher_suffix=teacher_suffix,
        teacher_type=teacher_type,
        model_type=model_type,
        template=template,
        mapping=mapping,
        alpha=0.1,
        gamma=0.1,
        temperature=15,
    )

    run_task(args)
