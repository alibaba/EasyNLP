# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD"""

import argparse
import glob
import logging
import os
import random
import timeit
import itertools
import json
import pickle
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset, ConcatDataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from collections import defaultdict
from prettytable import PrettyTable

from src.data.qa import squad_convert_examples_to_features
from src.bert import BertForQuestionAnswering
from src.xlmr import XLMRobertaForQuestionAnswering

from transformers import (
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    BertConfig,
    BertTokenizer,
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup
)

from src.data.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from src.data.qa import (
    SquadResult,
    MLQAProcessor,
    TydiqaProcessor,
    XquadProcessor
)
from src.data.mlqa_evaluation_v1 import evaluate_with_path

from src.utils import BatchContinuousRandomSampler, BatchContinuousDistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForQuestionAnswering, XLMRobertaTokenizer),
}

PROCESSOR_MAP = {
    "mlqa": MLQAProcessor,
    "tydiqa": TydiqaProcessor,
    "xquad": XquadProcessor,
}

LANG_2_IDX = {}

class AlignDataset(Dataset):
    def __init__(self, datasets, examples, feature_qas_ids, en_index):
        super(AlignDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'

        self.datasets = datasets
        self.en_index = en_index

        self.eng_id2feat = {}
        
        for idx, qas_id in enumerate(feature_qas_ids[en_index]):
            if qas_id not in self.eng_id2feat:
                self.eng_id2feat[qas_id] = []
            self.eng_id2feat[qas_id].append(idx)

        self.indexes = []
        qas_id_cnt = defaultdict(int)
        for lang_idx, qas_ids in enumerate(feature_qas_ids):
            for feat_idx, qas_id in enumerate(qas_ids):

                if lang_idx == en_index:
                    self.indexes.append((lang_idx, feat_idx, qas_id))
                else:
                    # for training, only add aligned data
                    if qas_id in self.eng_id2feat:
                        self.indexes.append((lang_idx, feat_idx, self.eng_id2feat[qas_id][qas_id_cnt[qas_id]]))
                        qas_id_cnt[qas_id] = (qas_id_cnt[qas_id] + 1 ) % len(self.eng_id2feat[qas_id])
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        lang_idx, feat_idx, idx3 = self.indexes[idx]

        if lang_idx == self.en_index:
            return [self.datasets[self.en_index][feat_idx], self.datasets[lang_idx][feat_idx]]
        else:
            # other languages
            return [self.datasets[self.en_index][idx3], self.datasets[lang_idx][feat_idx]]


class AlignDatasetForEval(Dataset):
    def __init__(self, datasets, feature_qas_ids):
        super(AlignDatasetForEval, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        assert len(datasets) == 2

        self.datasets = datasets
        self.indexes = []

        self.en_qas_id_2_idx = {}

        for idx, qas_id in enumerate(feature_qas_ids[0]):
            self.en_qas_id_2_idx[qas_id] = idx
        
        for idx, qas_id in enumerate(feature_qas_ids[1]):
            # TODO
            assert qas_id in self.en_qas_id_2_idx, "{} need English translation for inference".format(qas_id)
            self.indexes.append([self.en_qas_id_2_idx[qas_id], idx])

    def __len__(self):
        return len(self.datasets[1])

    def __getitem__(self, idx):
        feat_idx_en, feat_idx = self.indexes[idx]
        return [self.datasets[0][feat_idx_en], self.datasets[1][feat_idx]]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(input):
    if isinstance(input, list):
        return [tensor.detach().cpu.tolist() for tensor in input]
    else:
        # tensor
        return input.detach().cpu().tolist()


def get_max_steps(output_dir):
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
    max_step = None
    for checkpoint in checkpoints:
        if len(checkpoint.split("-")) > 1:
            if max_step is None:
                max_step = int(checkpoint.split("-")[-1])
            else:
                max_step = max(max_step, int(checkpoint.split("-")[-1]))
    return max_step


def save_model(args, model, tokenizer):
    # Create output directory if needed

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        if args.log_dir:
            tb_writer = SummaryWriter(args.log_dir)
        else:
            tb_writer = SummaryWriter()
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'a')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.norm:
        train_sampler = BatchContinuousRandomSampler(train_dataset, batch_size=args.per_gpu_train_batch_size) if args.local_rank == -1 else BatchContinuousDistributedSampler(train_dataset, batch_size=args.per_gpu_train_batch_size)
    else:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else math.ceil(
        args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    tr_loss, logging_loss, best_dev_score = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    # Added here for reproductibility
    set_seed(args)

    train_lang_ids = []
    train_langs = args.language.split(',')

    for lang in train_langs:
        train_lang_ids.append(LANG_2_IDX[lang])

    for epc in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], mininterval=10)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            input_ids = torch.stack([d[0] for d in batch], dim=1).to(args.device)
            attention_mask = torch.stack([d[1] for d in batch], dim=1).to(args.device)
            token_type_ids = torch.stack([d[2] for d in batch], dim=1).to(args.device)
            start_positions = torch.stack([d[3] for d in batch], dim=1).to(args.device)
            end_positions = torch.stack([d[4] for d in batch], dim=1).to(args.device)
            query_len = torch.stack([d[-2] for d in batch], dim=1).to(args.device)
            lang_ids = torch.stack([d[-1] for d in batch], dim=1).to(args.device)

            inputs = {"input_ids": input_ids, 
                        "token_type_ids": token_type_ids, 
                        "attention_mask": attention_mask, 
                        "start_positions":start_positions, 
                        "end_positions":end_positions,
                        "query_len": query_len,
                        "lang_ids": lang_ids}

            if args.model_type in ["xlmr"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                    logging_loss = tr_loss
                
                # eval and save model checkpoint
                if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step % args.eval_steps == 0 and global_step >= t_total * 0.3:
                    result = evaluate(args, model, tokenizer, prefix="", split='dev')
                    tb_writer.add_scalar(
                        "eval_exact", result['dev_avg']['exact_match'], global_step)
                    tb_writer.add_scalar(
                        "eval_f1", result['dev_avg']['f1'], global_step)
                    dev_score = (result['dev_avg']['exact_match'] + 2 * result['dev_avg']['f1']) / 3

                    log_writer.write("{0}\t{1}\n".format(global_step, json.dumps(result)))
                    log_writer.flush()
                    logger.info(result)

                    if dev_score >= best_dev_score:
                        best_dev_score = dev_score
                        save_model(args, model, tokenizer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # eval
    if args.local_rank in [-1, 0]:
        result = evaluate(args, model, tokenizer, prefix="", split='dev')
        tb_writer.add_scalar(
            "eval_exact", result['dev_avg']['exact_match'], global_step)
        tb_writer.add_scalar(
            "eval_f1", result['dev_avg']['f1'], global_step)
        dev_score = (result['dev_avg']['exact_match'] + result['dev_avg']['f1']) / 2

        log_writer.write("{0}\t{1}\n".format(global_step, json.dumps(result)))
        log_writer.flush()
        logger.info(result)

        if dev_score >= best_dev_score:
            best_dev_score = dev_score
            save_model(args, model, tokenizer)

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        log_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, prefix="", split='dev'):
    languages = list(args.language.split(','))

    if args.task_name == 'xquad' and split == 'dev':
        # languages in mlqa
        languages = ['en', 'ar', 'de', 'es', 'hi', 'vi', 'zh']
    elif args.task_name == 'tydiqa' and split == 'dev':
        # only use en for dev set
        languages = ['en']

    all_languages_results = {}
    processor = PROCESSOR_MAP[args.task_name]()

    args.eval_batch_size = args.per_gpu_eval_batch_size
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    for lang in tqdm(languages, desc="Evaluating"):

        logger.info("evaluating on {0} {1}".format(split, lang))

        
        dataset, examples, features = load_and_cache_examples(args, tokenizer, language=lang, split=split, output_examples=True)

        if args.task_name == 'tydiqa':
            dataset_en, examples_en, features_en = load_and_cache_examples(args, tokenizer, language=lang, split='translate-test', output_examples=True)
        else:
            dataset_en, examples_en, features_en = load_and_cache_examples(args, tokenizer, language='en', split=split, output_examples=True)

        feature_qas_ids = []
        feature_qas_ids.append([feature.qas_id for feature in features_en])
        feature_qas_ids.append([feature.qas_id for feature in features])

        dataset = AlignDatasetForEval([dataset_en, dataset], feature_qas_ids)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        all_results = []

        for batch in eval_dataloader:
            model.eval()

            input_ids = torch.stack([d[0] for d in batch], dim=1).to(args.device)
            attention_mask = torch.stack([d[1] for d in batch], dim=1).to(args.device)
            token_type_ids = torch.stack([d[2] for d in batch], dim=1).to(args.device)
            lang_ids = torch.stack([d[-1] for d in batch], dim=1).to(args.device)


            inputs = {"input_ids": input_ids, 
                        "token_type_ids": token_type_ids, 
                        "attention_mask": attention_mask,
                        "lang_ids": lang_ids}

            example_indices = batch[1][3]

            with torch.no_grad():
                if args.model_type in ["xlmr"]:
                    del inputs["token_type_ids"]

                outputs = model(**inputs)
                outputs = [outputs[0], outputs[1]] # [start_logits, end_logits]

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "pred_{}_{}_{}.json".format(prefix, split, lang))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}_{}.json".format(prefix, split, lang))

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}_{}_{}.json".format(prefix, split, lang))
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if args.model_type in ["xlnet", "xlm"]:
            start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
            end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                args.version_2_with_negative,
                tokenizer,
                args.verbose_logging,
            )
        else:
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                args.verbose_logging,
                args.version_2_with_negative,
                args.null_score_diff_threshold,
                tokenizer,
                map_to_origin=not (args.model_type == "xlmr" and (lang == 'zh' or lang == 'ko'))
            )

        # Compute the F1 and exact scores.
        if args.task_name in ['xquad', 'tydiqa']:
            results = squad_evaluate(examples, predictions)
        elif args.task_name == 'mlqa':
            results = evaluate_with_path(processor.get_dataset_path(args.data_dir, split, lang), output_prediction_file, lang)
        else:
            raise ValueError("not support yet")
        all_languages_results["{0}_{1}".format(split, lang)] = {'exact_match': results['exact_match'], 'f1': results['f1']}

    table = PrettyTable()
    table.title = f"{args.task_name}-{split}"
    table.add_column('lang', ['EM', 'F1'])
    for lang in languages:
        table.add_column(lang, [
            '%.2f' % (all_languages_results[f"{split}_{lang}"]['exact_match']),
            '%.2f' % (all_languages_results[f"{split}_{lang}"]['f1']),
        ])

    table.add_column('Avg.', [
        '%.2f' % np.mean([all_languages_results[f"{split}_{lang}"]['exact_match'] for lang in languages]),
        '%.2f' % np.mean([all_languages_results[f"{split}_{lang}"]['f1'] for lang in languages])
    ])

    logger.info(table)

    all_languages_results["{0}_avg".format(split)] = average_dic([value for key, value in all_languages_results.items() if split in key])

    return all_languages_results

def average_dic(dic_list):
    if len(dic_list) == 0:
        return {}
    dic_sum = {}
    for dic in dic_list:
        if len(dic_sum) == 0:
            for key, value in dic.items():
                dic_sum[key] = value
        else:
            assert set(dic_sum.keys()) == set(dic.keys()), "sum_keys:{0}, dic_keys:{1}".format(set(dic_sum.keys()), set(dic.keys()))
            for key, value in dic.items():
                dic_sum[key] += value
    for key in dic_sum:
        dic_sum[key] /= len(dic_list)
    return dic_sum

def filter_examples(examples, max_num, examples_dev=None):
    exist_ids = {}
    new_examples = []
    if examples_dev is not None:
        for example in examples_dev:
            if example.qas_id not in exist_ids:
                exist_ids[example.qas_id] = 1
        
        for example in examples:
            if example.qas_id in exist_ids:
                new_examples.append(example)
        
    else:
        for example in examples:
            if example.qas_id not in exist_ids:
                exist_ids[example.qas_id] = 1
                new_examples.append(example)
    
    return new_examples[:max_num]

def load_and_cache_examples(args, tokenizer, language, split="train", output_examples=False, use_barrier=True):

    if use_barrier and args.local_rank not in [-1, 0] and split == "train":
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.cache_dir if args.cache_dir else "."
    data_cache_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    if args.data_cache_name is not None:
        data_cache_name = args.data_cache_name
    
    if args.task_name in ['mlqa', 'xquad'] and split == 'train':
        task_name = 'squad'
    elif args.task_name == 'xquad' and split == 'dev':
        task_name = 'mlqa'
    else:
        task_name = args.task_name
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}_{}".format(
            task_name,
            split,
            language,
            data_cache_name,
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s, language %s", input_dir, language)

        if not args.data_dir:
            raise ValueError("data dir can't be empty")
        processor = PROCESSOR_MAP[args.task_name]()
        if split == "dev":
            examples = processor.get_dev_examples_by_language(args.data_dir, language=language)
        elif split == "test":
            examples = processor.get_test_examples_by_language(args.data_dir, language=language)
        elif split == "translate-test":
            examples = processor.get_translate_test_examples_by_language(args.data_dir, language=language)
        else:
            examples = processor.get_train_examples_by_language(args.data_dir, language=language)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=split=="train",
            return_dataset="pt",
            threads=args.threads,
            lang_id=LANG_2_IDX['en'] if split in ['translate-test', 'translate-dev'] else LANG_2_IDX[language],
        )

        if args.local_rank in [-1, 0]:
            os.makedirs(input_dir, exist_ok=True)
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if use_barrier and args.local_rank == 0 and split=="train":
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        # help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--data_cache_name",
        default=None,
        type=str,
        help="The name of cached data",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="The output log dir."
    )

    parser.add_argument(
        "--benchmark", default='xtreme', type=str, choices=['xglue', 'xtreme'], help="xglue/xtreme"
    )
    parser.add_argument(
        "--task_name", default='mlqa', type=str, help="task"
    )
    parser.add_argument(
        "--pkl_index", default="0", type=str, help="pickle index for teach student training"
    )
    parser.add_argument(
        "--use_squad_for_tydiqa", action='store_true', help="include squad english data for tydiqa training"
    )
    parser.add_argument(
        "--gpu_id", default=None, type=str, help="GPU id"
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        default=0.1,
        type=float,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_samples_per_epoch", default=None, type=int, help="Not use, for consistent usage with classification and tagging tasks"
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--eval_steps", type=int, default=200, help="Eval every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_training", action="store_true", help="Overwrite the cached training model"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    # cross-lingual part
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Train and evaluation language.",
    )

    parser.add_argument("--eval_split", default='test', type=str, help="")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--mix_layer", default=7, type=int)
    parser.add_argument("--mix_layers", default=None, type=str)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--teaching_weight", default=0.1, type=float)
    parser.add_argument("--consist_weight", default=0.1, type=float)
    parser.add_argument("--align_weight", default=0.01, type=float)
    parser.add_argument("--temp", default=0.05, type=float)
    parser.add_argument("--cl", action="store_true")


    args = parser.parse_args()

    train_langs_t = args.language.split(',')
    train_langs = []

    for l in train_langs_t:
        if l not in train_langs:
            train_langs.append(l)

    for i, lang in enumerate(train_langs):
        LANG_2_IDX[lang] = i

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()


    # Setup CUDA, GPU & distributed training
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    if args.local_rank in [-1, 0]:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path
    )


    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        num_lang=len(args.language.split(',')),
        args=args
    )


    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        train_langs = args.language.split(',')

        if args.local_rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        logger.info(train_langs)

        datasets, feature_qas_ids = [], []
        examples, features = [], []
        for lang in train_langs:

            lg_train_dataset, lg_train_examples, lg_train_features = load_and_cache_examples(args,
                                                                                             tokenizer,
                                                                                             language=lang,
                                                                                             split="train",
                                                                                             output_examples=True,
                                                                                             use_barrier=True)
            datasets.append(lg_train_dataset)
            examples.append(lg_train_examples)
            features.append(lg_train_features)
            feature_qas_ids.append([feature.qas_id for feature in lg_train_features])

        train_dataset = AlignDataset(datasets, examples, feature_qas_ids, en_index=train_langs.index('en'))

        if args.local_rank == 0:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'a')

        # Load model from output_dir
        checkpoints = [args.output_dir]

        results = {}
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(
                checkpoint,
                config=config,
                num_lang=len(args.language.split(',')),
                args=args
            )
            model.to(args.device)

            result = evaluate(args, model, tokenizer, prefix=global_step, split=args.eval_split)

            filtered_result = {}
            for k, v in result.items():
                filtered_result[k] = { key:val for key, val in v.items() if key in ['exact', 'exact_match', 'f1']}
            log_writer.write("{}\t{}\n".format(checkpoint, json.dumps(filtered_result)))

            results[checkpoint] = filtered_result

        log_writer.close()
        logger.info("Results: {}".format(results))


    logger.info("Task Finished!")

if __name__ == "__main__":
    main()
