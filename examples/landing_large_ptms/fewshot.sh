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

wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/train.csv
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/dev.csv

easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./fewshot_model/ \
    --learning_rate=1e-5 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=512 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
        enable_fewshot=True
        label_desc=好,差
        type=pet_fewshot
        pattern=text,是一条商品,label,评。
    "