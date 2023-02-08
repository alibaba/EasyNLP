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

from megatron import get_args
from megatron.model import ModelType
from rapidformer.application.text_generation.gpt_predictor import GPTPredictor
from rapidformer.engine.engine import RapidformerEngine
from rapidformer.model.transformer.gpt_model import GPTModel


def get_tasks_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument('--temperature',
                       type=float,
                       default=1.0,
                       help='Sampling temperature.')
    group.add_argument('--top_p',
                       type=float,
                       default=0.0,
                       help='Top p sampling.')
    group.add_argument('--top_k', type=int, default=0, help='Top k sampling.')
    group.add_argument('--out-seq-length',
                       type=int,
                       default=1024,
                       help='Size of the output generated text.')
    group.add_argument('--text-generate-input-file',
                       type=str,
                       default='./sample.txt',
                       help='1762 samples to test text generate performence')
    group.add_argument('--text-generate-output-file',
                       type=str,
                       default='',
                       help='result of 1762 samples text')
    group.add_argument('--time',
                       action='store_true',
                       help='measure end to end text generation average time')
    group.add_argument(
        '--input_len',
        type=int,
        default=1,
        help='input lenth for measure end to end text generation average time')
    return parser


class MegatronGPTMoEPredictor(GPTPredictor):
    def __init__(
        self,
        engine,
    ):
        super().__init__(engine=engine)

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


if __name__ == '__main__':
    engine = RapidformerEngine(extra_args_provider=get_tasks_args)
    predictor = MegatronGPTMoEPredictor(engine=engine)
    predictor.predict()
