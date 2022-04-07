#!/bin/bash

if [ $# -lt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$1
fi

WORKER_COUNT=1
WORKER_GPU=1

if [ ! -f ./dev.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/m6/sentence_classification/dev.tsv
fi

echo '=========[ Augment news data with Chinese RoBERTa ]========='
easynlp \
    --app_name=data_augmentation \
    --worker_count=${WORKER_COUNT} \
    --worker_gpu=${WORKER_GPU} \
    --mode=predict \
    --tables=dev.tsv \
    --input_schema=index:str:1,sent:str:1,label:str:1 \
    --first_sequence=sent \
    --label_name=label \
    --outputs=aug.tsv \
    --output_schema=augmented_data \
    --checkpoint_dir=_ \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
        type=mlm_da
        expansion_rate=2
        mask_proportion=0.1
        remove_blanks=True
    "
