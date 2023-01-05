#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=5001
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=$1
MODEL_SIZE=$2
MOE=$3
TP=$4
BS=$5
SEQ_LEN=$6
TOP_K=$7
INPUT_SEQ_LEN=$8
OUTPUT_SEQ_LEN=$9
INPUT_FILE=${10}
OUTPUT_FILE=${11}
TIME=${12}

if [ ! -f tokenizer.json ]; then
  wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
fi

tokenizer_options=" \
    --tokenizer-type JiebaBPETokenizer \
    --vocab-file tokenizer.json
    "

if [ $MODEL_SIZE = 0.125B ]; then

NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12

elif [ $MODEL_SIZE = 0.35B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16

elif [ $MODEL_SIZE = 0.76B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=16

elif [ $MODEL_SIZE = 1.3B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32

elif [ $MODEL_SIZE = 2.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=2560
NUM_ATTN_HEADS=32

elif [ $MODEL_SIZE = 3.6B ]; then

NUM_LAYERS=30
HIDDEN_SIZE=3072
NUM_ATTN_HEADS=32

elif [ $MODEL_SIZE = 6.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32


fi

if [ $TIME = none ]; then
    time_options=" \
		               "
else
    time_options=" \
        --time "
fi

if [ $INPUT_FILE = none ]; then
    input_options=" \
		               "
else
    input_options=" \
        --text-generate-output-file ${OUTPUT_FILE}\
        --text-generate-input-file ${INPUT_FILE} \
        "
fi


if [ $MOE = none ]; then
    moe_options=" \
		               "
else
    moe_options=" \
        --disable-moe-token-dropping \
        --moe-min-capacity 0 \
        --num-experts ${MOE}
                    "
fi


rapidformer_options="  \
       --pretrained-model-name-or-path ${CHECKPOINT_PATH} \
       --micro-batch-size ${BS} \
       --num-layers ${NUM_LAYERS}  \
       --hidden-size ${HIDDEN_SIZE}  \
       --num-attention-heads ${NUM_ATTN_HEADS}  \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --no-load-optim \
       --no-load-rng \
       --inference \
       --temperature 1.0  \
       --top_k ${TOP_K} \
       --input_len ${INPUT_SEQ_LEN} \
       --out-seq-length ${OUTPUT_SEQ_LEN}  \
       --tensor-model-parallel-size ${TP} \
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS generate_text_gpt.py ${tokenizer_options}
 ${rapidformer_options} ${moe_options} ${time_options} ${input_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
