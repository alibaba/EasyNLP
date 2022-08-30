#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=7
MASTER_ADDR=localhost
MASTER_PORT=6011
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

PRETRAINED_CHECKPOINT='bert-base-uncased'

common_cmd="--task sequence_classification \
            --app-name=text_classify \
            --user-defined-parameters='pretrain_model_name_or_path=bert-base-uncased' \
            --pretrained-model-name-or-path $PRETRAINED_CHECKPOINT \
            --data-dir glue \
            --data-name mrpc \
            --micro-batch-size 16 \
            --global-batch-size 16 \
            --epochs 3 \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-position-embeddings 512 \
            --seq-length 512 \
            --lr 2e-5 \
            --lr-decay-style linear \
            --lr-warmup-iters 100 \
            --weight-decay 1e-2 \
            --clip-grad 1.0 \
            --seed 42 \
            --log-interval 100 \
            --eval-interval 1000 \

"

CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch $DISTRIBUTED_ARGS rf_finetune_easynlp_bert.py \
       $common_cmd \
       --mixed-precision




