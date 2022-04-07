#!/bin/bash
set -e

if [ $# -lt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

WORKER_COUNT=1
WORKER_GPU=1

if [ ! -f ./ner_train.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/knowledge_nlu/ner_train.csv
fi

if [ ! -f ./ner_dev.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/knowledge_nlu/ner_dev.csv
fi

echo '=========[ Training: Medical NER ]========='

easynlp \
    --mode=train \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=ner_train.csv,ner_dev.csv \
    --input_schema=content:str:1,ner_tags:str:1 \
    --first_sequence=content \
    --label_name=ner_tags \
    --label_enumerate_values=O,B-test,I-test,B-disease,I-disease,B-physiology,I-physiology,B-body,I-body,B-feature,I-feature,B-department,I-department,B-drug,I-drug,B-crowd,I-crowd,B-treatment,I-treatment,B-symptom,I-symptom,B-time,I-time \
    --checkpoint_dir=./ner_model \
    --learning_rate=3e-5  \
    --epoch_num=3  \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=512 \
    --micro_batch_size=8 \
    --app_name=sequence_labeling \
    --user_defined_parameters="pretrain_model_name_or_path=alibaba-pai/pai-dkplm-medical-base-zh"
    
echo '=========[ Evaluation: Medical NER ]========='

easynlp \
      --mode=evaluate \
      --worker_count=${WORKER_COUNT} \
      --worker_gpu=${WORKER_GPU} \
      --tables=ner_dev.csv \
      --input_schema=content:str:1,ner_tags:str:1 \
      --first_sequence=content \
      --label_name=ner_tags \
      --label_enumerate_values=O,B-test,I-test,B-disease,I-disease,B-physiology,I-physiology,B-body,I-body,B-feature,I-feature,B-department,I-department,B-drug,I-drug,B-crowd,I-crowd,B-treatment,I-treatment,B-symptom,I-symptom,B-time,I-time \
      --checkpoint_dir=./ner_model \
      --sequence_length=512 \
      --micro_batch_size=8 \
      --app_name=sequence_labeling
      
echo '=========[ Prediction: Medical NER ]========='

easynlp \
    --mode=predict \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --tables=ner_dev.csv \
    --outputs=ner_dev.pred.csv \
    --input_schema=content:str:1,ner_tags:str:1 \
    --output_schema=predictions \
    --append_cols=content,ner_tags \
    --first_sequence=content \
    --label_name=ner_tags \
    --checkpoint_dir=./ner_model \
    --micro_batch_size=8 \
    --sequence_length=512 \
    --app_name=sequence_labeling