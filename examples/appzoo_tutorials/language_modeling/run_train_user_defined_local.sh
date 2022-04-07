#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train.json ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/train.json
fi

if [ ! -f ./dev.json ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dev.json
fi

MASTER_ADDR=localhost
MASTER_PORT=6009
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=train \
    --worker_gpu=1 \
    --tables=train.json,dev.json \
    --learning_rate=1e-4  \
    --epoch_num=3  \
    --logging_steps=100 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --train_batch_size=2 \
    --checkpoint_dir=./lm_models \
    --app_name=language_modeling \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
    '

