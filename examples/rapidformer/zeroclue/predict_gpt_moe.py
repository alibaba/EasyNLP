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
from megatron import get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids

from rapidformer import RapidformerEngine, get_args, GPTPredictor
from rapidformer.model_adapter.parallel_transformer import GPTModel

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='gpt moe')

    group.add_argument('--predict-output-dir',  type=str, default=None, help="column names for input files")

    return parser

class MegatronGPTMoEPredictor(GPTPredictor):
    def __init__(self,
                 engine,
                 ):
        super().__init__(engine=engine)

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
        args = get_args()
        tokenizer = get_tokenizer()

        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens = data_b['text'].long()

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        output_tensor, *other_losses = model(tokens, position_ids, attention_mask, tokentype_ids=None,
                        forward_method_parallel_output=False)

        return output_tensor

if __name__ == "__main__":
    engine = RapidformerEngine(extra_args_provider=get_tasks_args)
    predictor = MegatronGPTMoEPredictor(engine=engine)
    predictor.predict()
