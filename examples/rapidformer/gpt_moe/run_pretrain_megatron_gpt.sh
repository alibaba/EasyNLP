#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=localhost
MASTER_PORT=6012
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

MODE=$1
TOKENIZER=$2
MODEL_SIZE=$3
MOE=$4
RT=$5
BATCH_SIZE=$6
TP=$7
AC=$8
ZERO=$9
SAVE_INTERVAL=${10}
SEQ_LEN=2048

if [ $TOKENIZER = gpt2bpe ]; then

  if [ ! -f gpt2-vocab.json ]; then
    wget https://easytransfer-new.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/gpt2-vocab.json
  fi

  if [ ! -f gpt2-merges.txt ]; then
    wget https://easytransfer-new.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/gpt2-merges.txt
  fi

  TRAIN_TOKENS=300000000000
  WARMUP_TOKENS=375000000
  DATASET=pile


  dataset_options=" \
		    --tokenizer-type GPT2BPETokenizer \
		    --vocab-file gpt2-vocab.json \
		    --merge-file gpt2-merges.txt \
		    --data-path /mnt/pile/pile_text_document
		    "

elif [ $TOKENIZER = jiebabpe ]; then

    if [ ! -f tokenizer.json ]; then
      wget https://easynlp-dev.oss-cn-zhangjiakou.aliyuncs.com/225247/RapidformerPro/tokenizer.json
    fi

    #wudao dataset train 2 epochs
    TRAIN_TOKENS=100000000000
    WARMUP_TOKENS=125000000
    DATASET=wudao

    dataset_options=" \
		    --tokenizer-type JiebaBPETokenizer \
		    --vocab-file tokenizer.json \
		    --data-path /mnt/wudao/wudao_jiebabpe_text_document
		    "
fi


if [ $MODEL_SIZE = 0.125B ]; then

NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
LR=6e-4
MIN_LR=6e-5
GLOBAL_BATCH_SIZE=$(( 524288 / ${SEQ_LEN} ))

elif [ $MODEL_SIZE = 0.35B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
LR=3e-4
MIN_LR=3e-5
GLOBAL_BATCH_SIZE=$(( 524288 / ${SEQ_LEN} ))

elif [ $MODEL_SIZE = 0.76B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=16
LR=2.5e-4
MIN_LR=2.5e-5
GLOBAL_BATCH_SIZE=$(( 524288 / ${SEQ_LEN} ))

elif [ $MODEL_SIZE = 1.3B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
LR=2e-4
MIN_LR=2e-5
GLOBAL_BATCH_SIZE=$(( 1048576 / ${SEQ_LEN} ))

elif [ $MODEL_SIZE = 2.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=2560
NUM_ATTN_HEADS=32
LR=1.6e-4
MIN_LR=1.6e-5
GLOBAL_BATCH_SIZE=$(( 1048576 / ${SEQ_LEN} ))

elif [ $MODEL_SIZE = 3.6B ]; then

NUM_LAYERS=30
HIDDEN_SIZE=3072
NUM_ATTN_HEADS=32
LR=1.6e-4
MIN_LR=1.6e-5
GLOBAL_BATCH_SIZE=$(( 1048576 / ${SEQ_LEN} ))

elif [ $MODEL_SIZE = 6.7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
LR=1.2e-4
MIN_LR=1.2e-5
GLOBAL_BATCH_SIZE=$(( 2097152 / ${SEQ_LEN} ))

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

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="${MODE}-${DATASET}-${TOKENIZER}-megatron-gpt-${model_type}-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-tp-${TP}-ac-${AC}-zero-${ZERO}"
OUTPUT_BASEPATH=/mnt/output_${DATASET}
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
LOGGING_PATH="${OUTPUT_BASEPATH}/log/${NAME}_${current_time}"

rapidformer_options=" \
        --pretrained-model-name-or-path ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --split 98,2,0 \
        --data-impl mmap \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --logging-path $LOGGING_PATH \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --sequence-parallel \
        --moe-linear-layer-type standard \
        --use-expert-residual-network \
        --fp16\
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_megatron_gpt.py ${dataset_options}
 ${rapidformer_options} ${activation_checkpoint_options} ${zero_options} ${moe_options}"


echo ${run_cmd}
eval ${run_cmd}
set +x
