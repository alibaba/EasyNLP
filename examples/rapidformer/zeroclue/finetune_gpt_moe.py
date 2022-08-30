# coding=utf-8
# Copyright (c) 2021 Alibaba PAI Team.
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
import deepspeed
import torch
from megatron import mpu
from functools import partial
from megatron.utils import average_losses_across_data_parallel_group
from megatron import get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids

from rapidformer import RapidformerEngine, get_args, GPTFinetuner
from rapidformer.model_adapter.parallel_transformer import GPTModel
from rapidformer.data.pet_dataset_finetune_gpt import PETDataset as Dataset

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='fewclue pet')

    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')

    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')

    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')

    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='Sliding window for overlapping evaluation.')

    group.add_argument('--strict-lambada', action='store_true',
                       help='Use more difficult formulation of lambada.')

    group.add_argument('--valid-batch-size', type=int, default=None,
                       help='valid-batch-size')

    group.add_argument('--input_schema',  type=str, default=None, help="column names for input files")
    group.add_argument('--pattern',  type=str, default=None, help="format of human-generated prompts")
    group.add_argument('--label_desc',  type=str, default=None, help="maps of the label")
    group.add_argument('--label_enumerate_values', type=str, default=None, help="label values")

    return parser

class MegatronGPTMoEFinetuner(GPTFinetuner):
    def __init__(self,
                 engine,
                 ):
        super().__init__(engine=engine)

    def train_valid_test_datasets_provider(self):
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()
        if args.label_enumerate_values is not None and args.label_desc is not None:
            try:
                label_map = dict(zip(args.label_enumerate_values.split(','), args.label_desc.split(',')))
            except:
                raise ValueError("lengths of label_enumerate_values and label_desc do not match")
        else:
            label_map = None

        train_dataset = Dataset('clue', 'training', args.train_data,
                                tokenizer, args.seq_length, args.input_schema, args.pattern, label_map)

        valid_dataset = Dataset('clue','validation', args.valid_data,
                                tokenizer, args.seq_length, args.input_schema, args.pattern, label_map)

        test_dataset = Dataset('clue','validation', args.valid_data,
                                tokenizer, args.seq_length, args.input_schema, args.pattern, label_map)

        return train_dataset, valid_dataset, test_dataset, None

    def model_optimizer_lr_scheduler_provider(self, pre_process=True, post_process=True):
        args = get_args()
        with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                 remote_device=None if args.remote_device == 'none' else args.remote_device,
                                 config_dict_or_path=args.deepspeed_configuration,
                                 enabled=args.zero_stage == 3,
                                 mpu=mpu):
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )

            return model, None, None

    def run_forward_step(self, data_iterator, model):
        """Forward step."""
        """Forward step."""
        args = get_args()
        tokenizer = get_tokenizer()
        tokens_ = data_iterator['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        output_tensor, *other_losses = model(tokens, position_ids, attention_mask,
                                             labels=labels)
        moe_losses = []
        for moe_loss in other_losses:
            if moe_loss is not None:
                moe_losses.append(moe_loss)
        moe_loss = sum(moe_losses) * args.moe_loss_coeff
        def loss_func(loss_mask, moe_loss, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            averaged_loss = average_losses_across_data_parallel_group([loss])
            if max(args.num_experts) <= 1:
                return loss, {'lm loss': averaged_loss[0]}
            else:
                loss = loss + moe_loss
                return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}

        return output_tensor, partial(loss_func, loss_mask, moe_loss)

if __name__ == "__main__":
    engine = RapidformerEngine(extra_args_provider=get_tasks_args)
    finetuner = MegatronGPTMoEFinetuner(engine=engine)
    finetuner.train()
