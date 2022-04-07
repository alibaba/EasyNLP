#!/bin/bash
set -e

if [ $# -lt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

WORKER_COUNT=1
WORKER_GPU=1

if [ ! -f ./fewshot_train.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fewshot_learning/fewshot_train.tsv  
fi

if [ ! -f ./fewshot_dev.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fewshot_learning/fewshot_dev.tsv
fi

echo '=========[ Fewshot Training: P-tuning on Text Classification ]========='
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=./fewshot_train.tsv,./fewshot_dev.tsv \
    --input_schema=sid:str:1,sent1:str:1,sent2:str:1,label:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./fewshot_model/ \
    --learning_rate=1e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=512 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
        enable_fewshot=True
        label_desc=否,能
        type=pet_fewshot
        pattern=sent1,<pseudo>,label,<pseudo>,sent2
    "

echo '=========[ Fewshot Prediction: P-tuning on Text Classification  ]========='
easynlp \
    --app_name=text_classify \
    --mode=predict \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=./fewshot_train.tsv \
    --outputs=pred.tsv \
    --output_schema=predictions \
    --input_schema=sid:str:1,sent1:str:1,sent2:str:1,label:str:1 \
    --worker_count=1 \
    --worker_gpu=1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --append_cols=sid,label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./fewshot_model/ \
    --micro_batch_size=8 \
    --sequence_length=512 \
    --user_defined_parameters="
        enable_fewshot=True
        label_desc=否,能
        type=pet_fewshot
        pattern=sent1,<pseudo>,label,<pseudo>,sent2
    "