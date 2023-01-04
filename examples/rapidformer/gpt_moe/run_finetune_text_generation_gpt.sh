#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_DATASETS_OFFLINE=0
MASTER_ADDR=localhost
MASTER_PORT=6034
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

TASK_NAME=$1
TRAIN_DATASET_PATH=$2
VALID_DATASET_PATH=$3
PRETRAIN_CHECKPOINT_PATH=$4
MODEL_SIZE=$5
MOE=$6
RT=$7
BATCH_SIZE=$8
EPOCH=$9
TP=${10}
AC=${11}
ZERO=${12}
SEQ_LEN=${13}


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
LR=6e-4
MIN_LR=6e-5

elif [ $MODEL_SIZE = 0.35B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
LR=3e-4
MIN_LR=3e-5


elif [ $MODEL_SIZE = 0.76B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=16
LR=2.5e-4
MIN_LR=2.5e-5


elif [ $MODEL_SIZE = 1.3B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
LR=3e-4
MIN_LR=2e-5


elif [ $MODEL_SIZE = 2.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=2560
NUM_ATTN_HEADS=32
LR=1.6e-4
MIN_LR=1.6e-5


elif [ $MODEL_SIZE = 3.6B ]; then

NUM_LAYERS=30
HIDDEN_SIZE=3072
NUM_ATTN_HEADS=32
LR=1.6e-4
MIN_LR=1.6e-5

elif [ $MODEL_SIZE = 6.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
LR=1.2e-4
MIN_LR=1.2e-5


fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $ZERO = do ]; then
    zero_options=" \
		    --use-distributed-optimizer"

elif [ $ZERO = none ]; then
    zero_options=" \
                    "
fi

if [ $MOE = none ]; then
    moe_options=" \
		               "
		model_type="dense"

else
    moe_options=" \
        --router-type ${RT} \
        --num-experts ${MOE}
                    "
    model_type="moe-${MOE}-rt-${RT}"

fi

FT_NAME="finetune-${TASK_NAME}-megatron-gpt-${model_type}-${MODEL_SIZE}-lr-${LR}-epo-${EPOCH}-bs-${BATCH_SIZE}-tp-${TP}-ac-${AC}-zero-${ZERO}"
OUTPUT_BASEPATH=/mnt/output_wudao
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${FT_NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}

FINETUNE_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${FT_NAME}"
LOGGING_PATH="${OUTPUT_BASEPATH}/log/${FT_NAME}_${current_time}"


rapidformer_options="  \
        --pretrained-model-name-or-path ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${FINETUNE_CHECKPOINT_PATH} \
        --train-data ${TRAIN_DATASET_PATH} \
        --valid-data ${VALID_DATASET_PATH} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --keep-last \
        --micro-batch-size ${BATCH_SIZE} \
        --epochs ${EPOCH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.01 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.006 \
        --num-workers 0\
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval 100000000 \
        --tensorboard-queue-size 1 \
        --logging-path $LOGGING_PATH \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --finetune \
        --no-load-optim\
        --DDP-impl local\
        --tensor-model-parallel-size ${TP} \
        --fp16
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_megatron_gpt.py ${tokenizer_options}
 ${rapidformer_options} ${activation_checkpoint_options} ${zero_options} ${moe_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
