# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Race."""
import os
import torch
from SwissArmyTransformer import mpu
import json
import functools
from .eval_utils import accuracy_func_provider
from easynlp.modelzoo.mg_utils.finetune_glm import finetune
from easynlp.modelzoo.mg_utils.pretrain_glm import get_batch
from collections import OrderedDict
from .dataset import Seq2SeqDataset, BlankLMDataset, ExtractionDataset
from .evaluate import rouge_metric, DecoderEvaluater, BlankLMEvaluater
# from tasks.superglue.evaluate import squad_exact_match, squad_f1

global_tokenizer = None


def seq2seq_forward_step(data, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    if timers is not None:
        timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    if timers is not None:
        timers('batch generator').stop()
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    # logits, loss_mask = logits[:, args.src_seq_length:], loss_mask[:, args.src_seq_length:]
    # target_ids = target_ids[:, args.src_seq_length:]
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    if args.label_smoothing > 0.0:
        epsilon = args.label_smoothing
        smooth_loss = -torch.nn.functional.log_softmax(logits, dim=-1).mean(dim=-1)
        losses = (1 - epsilon) * losses + epsilon * smooth_loss
    loss_mask = loss_mask.reshape(-1)
    # The loss is not normalized for fair comparison
    loss = torch.sum(losses.reshape(-1) * loss_mask) / loss_mask.sum()
    return loss, mems, 'bert'


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    if args.task.lower() == 'blank':
        train_dataset = BlankLMDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    elif args.task.lower() == 'extraction':
        train_dataset = ExtractionDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    else:
        train_dataset = Seq2SeqDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    global global_tokenizer
    global_tokenizer = tokenizer
    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Provide metrics callback function."""

    def single_dataset_provider(split):
        if args.task.lower() == 'blank':
            return BlankLMDataset(args, split=split, tokenizer=tokenizer)
        elif args.task.lower() == 'extraction':
            return ExtractionDataset(args, split=split, tokenizer=tokenizer)
        else:
            return Seq2SeqDataset(args, split=split, tokenizer=tokenizer)

    if args.task.lower() in ['blank', 'extraction']:
        evaluater = BlankLMEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        metric_dict = {}
    else:
        evaluater = DecoderEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        if args.tokenizer_type == "BertWordPieceTokenizer":
            dataset = 'cnn_dm'
        elif args.task.lower() == 'gigaword':
            dataset = 'gigaword'
        else:
            dataset = 'cnn_dm_org'
        if args.task.lower() in ['squad', 'squad_v1']:
            metric_dict = {"EM": squad_exact_match, "F1": squad_f1}
        else:
            metric_dict = OrderedDict({"rouge-1": functools.partial(rouge_metric, metric="rouge-1", dataset=dataset),
                                       "rouge-2": functools.partial(rouge_metric, metric="rouge-2", dataset=dataset),
                                       "rouge-l": functools.partial(rouge_metric, metric="rouge-l", dataset=dataset)})

    def output_func(predictions, examples, output_file):
        if args.task.lower() in ['chinesegen']:
            # output_path = os.path.join(args.log_dir, args.outputs)
            with open(args.outputs, "w", encoding='utf-8') as output:
                for prediction, example in zip(predictions, examples):
                    write_content = [prediction[0]]
                    write_content.append('||'.join(prediction))
                    for name in example.meta.keys():
                        if name in args.append_cols.split(','):
                            write_content.append(example.meta[name])
                    output.write('\t'.join(write_content) + '\n')
            return
        if args.task.lower() in ['squad', 'squad_v1']:
            with open(output_file, "w", encoding='utf-8') as output:
                res = {}
                for prediction, example in zip(predictions, examples):
                    idx = example.idx
                    if prediction.lower().replace(' ', '') == 'n/a':
                        prediction = ''
                    if idx not in res or res[idx] == '':
                        res[idx] = prediction
                json.dump(res, output)
            with open(output_file + ".refs", "w", encoding='utf-8') as output:
                for prediction, example in zip(predictions, examples):
                    res = {'id': example.idx, 'pred': prediction, 'gold': example.meta['answers']}
                    output.write(json.dumps(res) + '\n')
            return
        with open(output_file + ".hyps", "w", encoding='utf-8') as output:
            for prediction in predictions:
                output.write(prediction)
                output.write("\n")
        with open(output_file + ".refs", "w", encoding='utf-8') as output:
            for example in examples:
                output.write(example.meta["ref"])
                output.write("\n")
        if args.task.lower() == 'squad_generation':
            with open(output_file + ".source", "w", encoding='utf-8') as output:
                for example in examples:
                    output.write(example.text_a.replace("\n", " ") + " Answer: " + example.meta["answer"])
                    output.write("\n")

    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False)


def args_mapping(args, user_defined_parameters):
    args.src_seq_length = args.sequence_length
    args.seq_length = args.sequence_length
    args.load_pretrained = args.pretrained_model_name_or_path
    args.save = args.checkpoint_dir
    args.epochs = args.epoch_num
    args.batch_size = args.micro_batch_size
    args.lr = args.learning_rate
    args.warmup = args.warmup_proportion
    args.num_beams = int(user_defined_parameters.get('num_beams', args.tgt_seq_length))
    args.tgt_seq_length = int(user_defined_parameters.get('max_decoder_length', args.tgt_seq_length))
    args.min_tgt_length = int(user_defined_parameters.get('min_decoder_length', args.min_tgt_length))
    args.no_repeat_ngram_size = int(user_defined_parameters.get('no_repeat_ngram_size', args.no_repeat_ngram_size))
    if args.mode == 'predict':
        args.num_return_sequences = int(user_defined_parameters.get('num_return_sequences', 1))
    else:
        args.num_return_sequences = 1

    # model args
    if os.path.exists(os.path.join(args.pretrained_model_name_or_path, 'config.json')):
        json_config = json.load(open(os.path.join(args.pretrained_model_name_or_path, 'config.json'), 'r'))
    elif os.path.exists(os.path.join(args.checkpoint_dir, 'config.json')):
        json_config = json.load(open(os.path.join(args.checkpoint_dir, 'config.json'), 'r'))
    args.num_layers = int(json_config.get('num_layers', args.num_layers))
    args.num_attention_heads = int(json_config.get('num_attention_heads', args.num_attention_heads))
    args.hidden_size = int(json_config.get('hidden_size', args.hidden_size))
    args.max_position_embeddings = int(json_config.get('max_position_embeddings', args.max_position_embeddings))
    args.tokenizer_type = json_config.get('tokenizer_type', args.tokenizer_type)

    return args

def main(args, user_defined_parameters):
    args = args_mapping(args, user_defined_parameters)
    if args.src_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.src_seq_length
    if args.task.lower() in ['cnn_dm', 'cnn_dm_original', 'gigaword', 'blank', 'squad_generation', 'xsum',
                             'squad', 'squad_v1', 'extraction', 'cmrc', 'chinesegen']:
        finetune(args, train_valid_datasets_provider, {}, end_of_epoch_callback_provider=metrics_func_provider,
                 forward_step=seq2seq_forward_step)
        # remove duplicate checkpoint files
        for path, dirs, files in os.walk(args.checkpoint_dir):
            for di in dirs:
                if di != 'best':
                    os.system('rm -rf %s' % os.path.join(path, di))
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpointed_iteration.txt')
        os.system('echo \'best\' > %s' % checkpoint_path)
    else:
        raise NotImplementedError(args.task)
