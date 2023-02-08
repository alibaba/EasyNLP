#!/bin/bash
set -e

if [ $# -lt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

if [ ! -f ./train.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
fi

if [ ! -f ./dev.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

WORKER_COUNT=1
WORKER_GPU=1

TEACHER_MODEL=bert-large-uncased
TEACHER_CKPT=results/large-sst2-teacher

STUDENT_MODEL=bert-small-uncased
STUDENT_CKPT=results/small-sst2-student

LOGITS_PATH=results/large-sst2-teacher

echo '=========[ Finetune the teacher model ]========='
easynlp \
     --app_name=text_match \
     --mode=train \
     --worker_count=${WORKER_COUNT} \
     --worker_gpu=${WORKER_GPU} \
     --tables=train.tsv,dev.tsv \
     --input_schema=label:str:1,id:str:1,id2:str:1,sent1:str:1,sent2:str:1 \
     --first_sequence=sent1 \
     --second_sequence=sent2 \
     --label_name=label \
     --label_enumerate_values=0,1 \
     --checkpoint_dir=${TEACHER_CKPT} \
     --learning_rate=3e-5 \
     --epoch_num=1 \
     --random_seed=42 \
     --save_checkpoint_steps=100 \
     --sequence_length=128 \
     --micro_batch_size=32 \
     --user_defined_parameters="pretrain_model_name_or_path=${TEACHER_MODEL}"

echo '=========[ Save the teacher logits ]========='
 easynlp \
     --mode=predict \
     --worker_count=${WORKER_COUNT} \
     --worker_gpu=${WORKER_GPU} \
     --tables=train.tsv \
     --outputs=${LOGITS_PATH}/pred.tsv \
     --input_schema=label:str:1,id:str:1,id2:str:1,sent1:str:1,sent2:str:1 \
     --output_schema=logits \
     --first_sequence=sent1 \
     --second_sequence=sent2 \
     --checkpoint_path=${TEACHER_CKPT} \
     --micro_batch_size=32 \
     --sequence_length=128 \
     --app_name=text_match

echo '=========[ Finetune the student model w/ KD ]========='
easynlp \
    --app_name=text_match \
    --mode=train \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=train.tsv,dev.tsv \
    --input_schema=label:str:1,id:str:1,id2:str:1,sent1:str:1,sent2:str:1,logits:float:2 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=${STUDENT_CKPT} \
    --learning_rate=3e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --save_checkpoint_steps=200 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --user_defined_parameters="
        pretrain_model_name_or_path=${STUDENT_MODEL}
        enable_distillation=True
        type=vanilla_kd
        logits_name=logits
        logits_saved_path=${LOGITS_PATH}/pred.tsv
        temperature=5
        alpha=0.2
    "

echo '=========[ Test the trained student model ]========='
easynlp \
    --mode=predict \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=train.tsv \
    --outputs=student_pred.tsv \
    --input_schema=label:str:1,id:str:1,id2:str:1,sent1:str:1,sent2:str:1 \
    --output_schema=predictions,probabilities,logits \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --checkpoint_path=${STUDENT_CKPT} \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --app_name=text_match
