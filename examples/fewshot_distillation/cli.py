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

"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from datetime import datetime
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    set_seed,
)

from src.arguments import (
    DynamicDataTrainingArguments,
    DynamicTrainingArguments,
    ModelArguments,
)
from src.dataset import DistillatoryFewShotDataset, FewShotDataset
from src.models import (
    BertCRSDistillStudent,
    BertCRSDistillTeacher,
    BertForPromptFinetuning,
    RobertaCRSDistillTeacher,
    RobertaCRSDistillStudent,
    RobertaForPromptFinetuning,
    BertDistillStudent,
    resize_token_type_embeddings,
)
from src.processors import (
    bound_mapping,
    compute_metrics_mapping,
    num_labels_mapping,
    output_modes_mapping,
)
from src.trainer import DistillatoryTrainer, Trainer

logger = logging.getLogger(__name__)


def get_args() -> Tuple[
    ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments
]:
    parser = HfArgumentParser(
        (ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "prompt" in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.do_debug:
        try:
            import pydevd_pycharm
        except ImportError:
            logger.error("Please install `pydevd_pycharm` for Pycharm debugging.")
            exit(-1)

        pydevd_pycharm.settrace(
            "localhost",
            port=4399,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )

    data_args.save_logit_dir = training_args.save_logit_dir

    return model_args, data_args, training_args


def main(model_args, data_args, training_args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split("\t")
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]
            logger.info(
                "Specify load the %d-th prompt: %s | %s"
                % (data_args.prompt_id, data_args.template, data_args.mapping)
            )
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[
                        : data_args.top_n_template
                    ]
                logger.info(
                    "Load top-%d templates from %s"
                    % (len(data_args.template_list), data_args.template_path)
                )

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info(
                        "Specify load the %d-th template: %s"
                        % (data_args.template_id, data_args.template)
                    )

            if data_args.mapping_path is not None:
                assert (
                    data_args.mapping_id is not None
                )  # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info(
                    "Specify using the %d-th mapping: %s"
                    % (data_args.mapping_id, data_args.mapping)
                )

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists."
        )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info(
            "Task name: {}, number of labels: {}, output mode: {}".format(
                data_args.task_name, num_labels, output_mode
            )
        )
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == "prompt-demo":
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            logger.info(
                "Automatically convert the template to GPT-3's in-context learning."
            )
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ""
            old_template = old_template.replace("*cls*", "")
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ""
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace(
                        "_{}*".format(sent_id),
                        "_{}*".format(sent_num + sent_num * instance_id + sent_id),
                    )
                # Replace mask
                sub_template = sub_template.replace(
                    "*mask*", "*labelx_{}*".format(instance_id)
                )
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template  # Put context at the end
                else:
                    new_template = (
                        sub_template + new_template
                    )  # Put context at the beginning
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ""
                    old_template = old_template.replace("*cls*", "")
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ""
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace(
                                "_{}*".format(sent_id),
                                "_{}*".format(sent_num + sent_num * label_id + sent_id),
                            )
                        # Replace mask
                        sub_template = sub_template.replace(
                            "*mask*", "*label_{}*".format(label_id)
                        )
                        new_template = new_template + sub_template
                    logger.info(
                        "| {} => {}".format(data_args.template_list[i], new_template)
                    )
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ""
                old_template = old_template.replace("*cls*", "")
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ""
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace(
                            "_{}".format(sent_id),
                            "_{}".format(sent_num + sent_num * label_id + sent_id),
                        )
                    # Replace mask
                    sub_template = sub_template.replace(
                        "*mask*", "*label_{}*".format(label_id)
                    )
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    config.__dict__.update(
        alpha=training_args.alpha,
        beta=training_args.beta,
        gamma=training_args.gamma,
        temperature=training_args.temperature,
    )   
    # 
    if "prompt" in model_args.few_shot_type:
        if config.model_type == "roberta":
            if training_args.export_embedding:
                model_fn = RobertaCRSDistillTeacher
            elif training_args.student_mode:
                model_fn = RobertaCRSDistillStudent
            else:
                model_fn = RobertaForPromptFinetuning
        elif config.model_type == "bert":
            if training_args.export_embedding:
                model_fn = BertCRSDistillTeacher
            elif training_args.student_mode:
                model_fn = BertCRSDistillStudent
            else:
                model_fn = BertForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == "finetune":
        if config.model_type == "roberta":
            model_fn = AutoModelForSequenceClassification
        elif config.model_type == "bert":
            model_fn = BertDistillStudent
            config.__dict__.update(  
                bert_tiny=False 
            )
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == "tiny":
        model_fn = BertDistillStudent
        config.__dict__.update(bert_tiny=True)
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    dataset_cls = (
        DistillatoryFewShotDataset if training_args.student_mode else FewShotDataset
    )

    # Get our special datasets.
    train_dataset = dataset_cls(
        data_args,
        tokenizer=tokenizer,
        mode="train",
        use_demo=("demo" in model_args.few_shot_type),
    )
    eval_dataset = (
        dataset_cls(
            data_args,
            tokenizer=tokenizer,
            mode="dev",
            use_demo=("demo" in model_args.few_shot_type),
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        dataset_cls(
            data_args,
            tokenizer=tokenizer,
            mode="test",
            use_demo=("demo" in model_args.few_shot_type),
        )
        if training_args.do_predict
        else None
    )

    set_seed(training_args.seed)

    if model_args.few_shot_type == "tiny":
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,

            cache_dir=model_args.cache_dir,
        )
    else:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == "bert":
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(
            model, new_num_types=10, random_segment=model_args.random_segment
        )

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = (
            torch.tensor(train_dataset.label_word_list).long().cuda()
        )
    if output_modes_mapping[data_args.task_name] == "regression":
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([data_args.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([data_args.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer_cls = (
        DistillatoryTrainer
        if training_args.teacher_mode or training_args.student_mode
        else Trainer
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)

        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(
                model_args, os.path.join(training_args.output_dir, "model_args.bin")
            )
            torch.save(
                data_args, os.path.join(training_args.output_dir, "data_args.bin")
            )

        # Reload the best checkpoint (for eval)
        model = model_fn.from_pretrained(training_args.output_dir)
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = (
                torch.tensor(train_dataset.label_word_list).long().cuda()
            )
        if output_modes_mapping[data_args.task_name] == "regression":
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

    # Evaluation
    final_result = {
        "time": str(datetime.today()),
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(
                eval_dataset.args.task_name
            )
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics

            output_eval_file = os.path.join(
                training_args.output_dir,
                f"eval_results_{eval_dataset.args.task_name}.txt",
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(
                        "***** Eval results {} *****".format(
                            eval_dataset.args.task_name
                        )
                    )
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[
                            eval_dataset.args.task_name + "_dev_" + key
                        ] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(
                    mnli_mm_data_args,
                    tokenizer=tokenizer,
                    mode="test",
                    use_demo=("demo" in model_args.few_shot_type),
                )
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(
                test_dataset.args.task_name
            )
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir,
                f"test_results_{test_dataset.args.task_name}.txt",
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(
                        "***** Test results {} *****".format(
                            test_dataset.args.task_name
                        )
                    )
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[
                            test_dataset.args.task_name + "_test_" + key
                        ] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape(
                        [test_dataset.num_sample, -1, num_logits]
                    ).mean(axis=0)
                    np.save(
                        os.path.join(
                            training_args.save_logit_dir,
                            "{}-{}-{}.npy".format(
                                test_dataset.task_name,
                                training_args.model_id,
                                training_args.array_id,
                            ),
                        ),
                        logits,
                    )

            test_results.update(test_result)

    if training_args.teacher_mode:
        trainer.compute_metrics = None
        if training_args.export_embedding:
            trainer.export_logits(dataset=train_dataset, filename="inter_embedding.pkl")
        else:
            trainer.export_logits(dataset=train_dataset, filename="cls_logits.pkl")
            

    with FileLock("log.lock"):
        with open("log", "a") as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if "evaluation_strategy" in final_result:
                final_result.pop("evaluation_strategy")
            f.write(str(final_result) + "\n")

    return eval_results


if __name__ == "__main__":
    main(*get_args())
