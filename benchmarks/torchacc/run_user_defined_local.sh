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

export CUDA_VISIBLE_DEVICES=0

# Use torchacc to train model
cur_path=/tmp/EasyNLP
cd ${cur_path}
if [ ! -f ./tmp/train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
xlarun \
  $DISTRIBUTED_ARGS main.py \
  --mode train \
  --tables=./tmp/train.tsv,./tmp/dev.tsv \
  --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
  --first_sequence=sent1 \
  --second_sequence=sent2 \
  --label_name=label \
  --label_enumerate_values=0,1 \
  --checkpoint_dir=./tmp/classification_model/ \
  --learning_rate=3e-5  \
  --epoch_num=2  \
  --random_seed=42 \
  --logging_steps=10 \
  --save_checkpoint_steps=50 \
  --sequence_length=128 \
  --micro_batch_size=200 \
  --app_name=text_classify \
  --data_threads=11 \
  --use_amp \
  --use_torchacc \
  --user_defined_parameters='pretrain_model_name_or_path=bert-small-uncased'
