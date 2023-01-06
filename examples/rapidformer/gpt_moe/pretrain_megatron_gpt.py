# Copyright (c) 2021 Alibaba PAI Team and Nvidia Megatron Team.
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

from functools import partial

import torch

from megatron import get_args, mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import ModelType
from megatron.utils import (average_losses_across_data_parallel_group,
                            get_ltor_masks_and_position_ids)
from rapidformer.engine.engine import RapidformerEngine
from rapidformer.engine.initialize import get_tokenizer
from rapidformer.model.transformer.gpt_model import GPTModel
from rapidformer.trainer.pretrainer import PreTrainer


class MegatronGPTPreTrainer(PreTrainer):
    def __init__(
        self,
        engine,
    ):
        super().__init__(engine=engine)

    def train_valid_test_datasets_provider(self, train_val_test_num_samples):
        args = get_args()
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))

        return train_ds, valid_ds, test_ds

    def model_optimizer_lr_scheduler_provider(self,
                                              pre_process=True,
                                              post_process=True):
        args = get_args()
        args.model_type = ModelType.encoder_or_decoder
        model = GPTModel(num_tokentypes=0,
                         parallel_output=True,
                         pre_process=pre_process,
                         post_process=post_process)
        return model, None, None

    def run_forward_step(self, data_iterator, model):
        args = get_args()
        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        attention_mask, loss_mask, position_ids =\
            get_ltor_masks_and_position_ids(tokens,
                                            tokenizer.eod,
                                            args.reset_position_ids,
                                            args.reset_attention_mask,
                                            args.eod_mask_loss)

        output_tensor, *other_losses = model(tokens,
                                             position_ids,
                                             attention_mask,
                                             labels=labels)
        if args.moe:
            moe_losses = []
            for moe_loss in other_losses:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            moe_loss = sum(moe_losses) * args.moe_loss_coeff

        def loss_func(loss_mask, moe_loss, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss =\
                average_losses_across_data_parallel_group([loss])

            args = get_args()

            if args.router_type == 0 and args.moe:
                loss = loss + moe_loss
            else:
                loss = loss

            if args.moe:
                return loss, {
                    'lm loss': averaged_loss[0],
                    'moe loss': moe_loss
                }
            else:
                return loss, {'lm loss': averaged_loss[0]}

        if args.moe:
            return output_tensor, partial(loss_func, loss_mask, moe_loss)
        else:
            return output_tensor, partial(loss_func, loss_mask, None)


if __name__ == '__main__':
    engine = RapidformerEngine()
    trainer = MegatronGPTPreTrainer(engine=engine)
    trainer.train()
