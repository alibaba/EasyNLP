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

if [ ! -f ./book_wiki_owtv2_small_text_sentence.bin ]; then
  wget https://easytransfer-new.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/book_wiki_owtv2_small_text_sentence.bin
  wget https://easytransfer-new.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/book_wiki_owtv2_small_text_sentence.idx
fi

if [ ! -f bert-en-uncased-vocab.txt ]; then
  wget https://easytransfer-new.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/bert-en-uncased-vocab.txt
fi

DATA_PATH=book_wiki_owtv2_small_text_sentence
PRETRAINED_CHECKPOINT='bert-base-uncased'

python -m torch.distributed.launch $DISTRIBUTED_ARGS rf_pretrain_easynlp_bert.py \
       --app-name=language_modeling \
       --user-defined-parameters='pretrain_model_name_or_path=bert-base-uncased' \
       --task pretraining \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --pretrained-model-name-or-path $PRETRAINED_CHECKPOINT \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --micro-batch-size 16 \
       --global-batch-size 32 \
       --seq-length 512 \
       --tokenizer-type BertWordPieceLowerCase \
       --max-position-embeddings 512 \
       --train-iters 100 \
       --data-path $DATA_PATH \
       --vocab-file bert-en-uncased-vocab.txt  \
       --data-impl mmap \
       --split 980,20 \
       --distributed-backend nccl \
       --lr 1e-3 \
       --lr-decay-style linear \
       --min-lr 0.0 \
       --lr-decay-iters 2000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --mixed-precision \
       --fsdp-memory-optimization