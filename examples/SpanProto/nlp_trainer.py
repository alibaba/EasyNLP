# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 2:51 pm.
# @Author  : JianingWang
# @File    : nlp_trainer.py
# !/usr/bin/env python
# coding=utf-8

import math
import os
os.environ["NCCL_DEBUG"] = "INFO"
import time
import torch
import numpy as np
from callback.ema import ExponentialMovingAveragingCallback
from callback.freeze import FreezeCallback
from callback.logger import LoggerCallback
from processor import processor_map
from transformers import CONFIG_MAPPING, AutoConfig, AutoTokenizer, HfArgumentParser, set_seed
from HFTrainer import HFTrainer
from transformers.trainer_utils import get_last_checkpoint
from transformers import EarlyStoppingCallback
from config import ModelArguments, DataTrainingArguments, TrainingArguments
from callback.mlflow import MLflowCallback
from tool.common import init_logger
from models import MODEL_CLASSES, TOKENIZER_CLASSES
import logging

logger = logging.getLogger(__name__)
torch.set_printoptions(precision=3, edgeitems=5, linewidth=160, sci_mode=False)

def main():
    # See all possible arguments or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(training_args.output_dir, list(filter(None, model_args.model_name_or_path.split('/')))[-1])
    os.makedirs(training_args.output_dir, exist_ok=True)
    # Setup logging
    log_file = os.path.join(training_args.output_dir,
                            f'{model_args.model_name_or_path.split(os.sep)[-1]}-{data_args.task_name}-{time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())}.log')
    log_level = training_args.get_process_log_level()
    init_logger(log_file, log_level, training_args.local_rank)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len([i for i in os.listdir(training_args.output_dir) if not i.endswith("log")]) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.task_name in processor_map:
        processor = processor_map[data_args.task_name](data_args, training_args, model_args)
    else:
        raise ValueError("task name unknown")
    # Load pretrained model and tokenizer
    # The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        'finetuning_task': data_args.task_name
    }

    # if 'mlm' not in data_args.task_type:
    if hasattr(processor, 'labels'):
        config_kwargs['num_labels'] = len(processor.labels)
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        if 'longformer' in model_args.model_name_or_path:
            config.sep_token_id = 102
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    tokenizer_class = TOKENIZER_CLASSES.get(model_args.model_type, AutoTokenizer)
    if model_args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    processor.set_tokenizer(tokenizer)
    config = processor.set_config(config)

    if data_args.task_type == 'autocls':
        model_class = build_cls_model(config)
    else:
        model_class = MODEL_CLASSES[data_args.task_type]
    if model_args.from_scratch:
        logger.info("Training new model from scratch")
        model = model_class.from_config(config)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=True
        )

    try:
        model.resize_token_embeddings(len(tokenizer))
    except:
        print("Fail to resize token embeddings.")

    tokenized_datasets = processor.get_tokenized_datasets()
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if 'test' not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets['test']
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    data_collator = processor.get_data_collator()
    if hasattr(processor, 'compute_metrics'):
        compute_metrics = processor.compute_metrics
    else:
        compute_metrics = None

    # Initialize our Trainer
    # --mlflow_location：mlflow tracker
    # --early_stopping_patience default None, --metric_for_best_model default eval_loss, --load_best_model_at_end,
    # --freeze_epochs, --freeze_keyword

    callbacks = [LoggerCallback]
    if data_args.mlflow_location or data_args.tracking_uri:
        mlflow_callback = MLflowCallback(model_args, data_args, training_args)
        callbacks.append(mlflow_callback)

    if model_args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience))

    if model_args.freeze_epochs:
        callbacks.append(FreezeCallback(freeze_epochs=model_args.freeze_epochs, freeze_keyword=model_args.freeze_keyword))

    if model_args.ema:
        callbacks.append(ExponentialMovingAveragingCallback(model_args.ema_decay))

    if training_args.do_predict_during_train:
        from callback.evaluate import DoPredictDuringTraining
        callbacks.append(DoPredictDuringTraining(test_dataset, processor))

    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        try:
            metrics = trainer.evaluate()

            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            if data_args.task_type == 'mlm':
                try:
                    perplexity = math.exp(metrics["eval_loss"])
                except OverflowError:
                    perplexity = float("inf")
                metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        except:
            logger.info("UNK problems ...")

    if training_args.do_predict and not training_args.do_predict_during_train:
        logger.info("*** Predict ***")
        if not data_args.keep_predict_labels:
            for l in ['labels', 'label']:
                if l in test_dataset.column_names:
                    test_dataset = test_dataset.remove_columns(l)
        prediction = trainer.predict(test_dataset, metric_key_prefix="predict")
        logits = prediction.predictions
        if data_args.keep_predict_labels:
            label_ids = prediction.label_ids
        if hasattr(processor, 'save_result'):
            if trainer.is_world_process_zero():
                if not data_args.keep_predict_labels:
                    processor.save_result(logits)
                else:
                    processor.save_result(logits, label_ids)
        elif hasattr(processor, 'compute_metrics'):

            try:
                metrics = trainer.evaluate()

                max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                    test_dataset)
                metrics["eval_samples"] = min(max_eval_samples, len(test_dataset))
                if data_args.task_type == 'mlm':
                    try:
                        perplexity = math.exp(metrics["eval_loss"])
                    except OverflowError:
                        perplexity = float("inf")
                    metrics["perplexity"] = perplexity

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)
            except:
                logger.info("UNK problems ...")

        else:
            predictions = np.argmax(logits, axis=1)
            output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {data_args.task_name} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = processor.labels[item]
                        writer.write(f"{index}\t{item}\n")
if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     logger.error('TRAINING ERROR')
    #     logger.error(str(e))
