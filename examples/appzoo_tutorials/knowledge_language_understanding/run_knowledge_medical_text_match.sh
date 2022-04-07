#!/bin/bash
set -e

if [ $# -lt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

WORKER_COUNT=1
WORKER_GPU=1
    
if [ ! -f ./nlu_train.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/knowledge_nlu/nlu_train.csv
fi

if [ ! -f ./nlu_dev.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/knowledge_nlu/nlu_dev.csv
fi


echo '=========[ Training: Medical Text Matching ]========='

easynlp \
    --mode=train \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=nlu_train.csv,nlu_dev.csv \
    --input_schema=label:str:1,text1:str:1,text2:str:1 \
    --first_sequence=text1 \
    --second_sequence=text2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./medical_model \
    --learning_rate=3e-5  \
    --epoch_num=1  \
    --random_seed=42 \
    --save_checkpoint_steps=50 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --app_name=text_match \
    --user_defined_parameters="pretrain_model_name_or_path=alibaba-pai/pai-dkplm-medical-base-zh"
    
echo '=========[ Evaluation: Medical Text Matching ]========='

easynlp \
      --mode=evaluate \
      --worker_count=${WORKER_COUNT} \
      --worker_gpu=${WORKER_GPU} \
      --tables=nlu_dev.csv \
      --input_schema=label:str:1,text1:str:1,text2:str:1 \
      --first_sequence=text1 \
      --second_sequence=text2 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --checkpoint_dir=./medical_model \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=text_match
      
echo '=========[ Prediction: Medical Text Matching ]========='

easynlp \
    --mode=predict \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=nlu_dev.csv \
    --outputs=nlu_dev.pred.csv \
    --input_schema=label:str:1,text1:str:1,text2:str:1 \
    --output_schema=predictions \
    --append_cols=text1,text2,label \
    --first_sequence=text1 \
    --second_sequence=text2 \
    --checkpoint_dir=./medical_model \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --app_name=text_match