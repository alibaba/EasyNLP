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

import torch.cuda

from easynlp.appzoo import ClassificationDataset
from easynlp.appzoo import get_application_predictor, get_application_model, get_application_evaluator
from easynlp.appzoo import get_application_model_for_evaluation
from easynlp.core import PredictorManager
from easynlp.core import Trainer
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import get_pretrain_model_path

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()

    predictor = get_application_predictor(
        app_name=args.app_name, model_dir=args.checkpoint_dir,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        sequence_length=args.sequence_length)
    predictor_manager = PredictorManager(
        predictor=predictor,
        input_file=args.tables.split(",")[-1],
        input_schema=args.input_schema,
        output_file=args.outputs,
        output_schema=args.output_schema,
        append_cols=args.append_cols,
        batch_size=args.micro_batch_size
    )
    predictor_manager.run()