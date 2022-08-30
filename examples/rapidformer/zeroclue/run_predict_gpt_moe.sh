#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_DATASETS_OFFLINE=0
MASTER_ADDR=localhost
MASTER_PORT=6021
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ ! -f pai_chinese_bpe.model ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/rapidformer/zeroclue/pai_chinese_bpe.model
fi

VOCAB_FILE=pai_chinese_bpe.model
TOKENIZER=ChineseBPETokenizer


DIR=`pwd`
SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2048
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
BATCH_SIZE=4
MP_SIZE=1
NUM_EXPERTS="32 32 32 32 32 32 32 32 32 32 64 64"
TASK=csldcp

CHECKPOINT_PATH=/mnt/finetune_${TASK}

rapidformer_options="  \
        --pretrained-model-name-or-path ${CHECKPOINT_PATH} \
        --tokenizer-type ${TOKENIZER} \
        --vocab-file ${VOCAB_FILE} \
        --tensor-model-parallel-size ${MP_SIZE} \
        --num-experts ${NUM_EXPERTS} \
        --micro-batch-size ${BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --no-load-optim \
        --no-load-rng \
        --inference \
        --disable-moe-token-dropping \
        --adaptive-seq-len\
        --eval-fp32\
        --top-k-linear-strategy normal \
        --checkpoint-activations \
        --eval-tasks ${TASK} \
        --mlp-type residual \
        --predict-output-dir zeroclue \
        --num-fewshot 0"

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS predict_megatron_gpt_moe.py ${rapidformer_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x


