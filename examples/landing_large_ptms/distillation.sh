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

# train teacher
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
    --checkpoint_dir=./teacher_model/ \
    --learning_rate=1e-5 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
    "

# data augmentation
easynlp \
    --app_name=data_augmentation \
    --worker_count=1 \
    --worker_gpu=1 \
    --mode=predict \
    --tables=train.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --outputs=aug.csv \
    --output_schema=augmented_data \
    --checkpoint_dir=_ \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
        type=mlm_da
        expansion_rate=10
        mask_proportion=0.25
        remove_blanks=True
    "
    
# forward teacher logits
easynlp \
    --mode=predict \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=aug.csv \
    --outputs=logits.csv \
    --input_schema=text:str:1,label:str:1 \
    --output_schema=logits \
    --first_sequence=text \
    --checkpoint_path=./teacher_model/ \
    --micro_batch_size=8 \
    --sequence_length=128 \
    --app_name=text_classify

# train student w/ KD
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=aug.csv,dev.csv \
    --input_schema=text:str:1,label:str:1,logits:float:2 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./student_model/ \
    --learning_rate=1e-4 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=alibaba-pai/pai-bert-tiny-zh
        enable_distillation=True
        type=vanilla_kd
        logits_name=logits
        logits_saved_path=logits.csv
        temperature=1
        alpha=0.5
    "

# train student w/o. KD
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
    --checkpoint_dir=./small_model_2/ \
    --learning_rate=1e-4 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=alibaba-pai/pai-bert-tiny-zh
"