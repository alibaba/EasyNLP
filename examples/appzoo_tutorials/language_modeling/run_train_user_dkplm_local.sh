#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train_corpus.txt ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/train_corpus.txt
fi

if [ ! -f ./dev_corpus.json ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/dev_corpus.txt
fi

if [ ! -f ./entity_emb.txt ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/entity_emb.txt
fi

if [ ! -f ./rel_emb.txt ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/rel_emb.txt
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
    --tables=train_corpus.txt,dev_corpus.txt \
    --learning_rate=1e-4  \
    --epoch_num=1  \
    --logging_steps=100 \
    --save_checkpoint_steps=500 \
    --sequence_length=128 \
    --train_batch_size=32 \
    --checkpoint_dir=./lm_models \
    --app_name=language_modeling \
    --user_defined_parameters='
        pretrain_model_name_or_path=alibaba-pai/pai-dkplm-medical-base-zh entity_emb_file=entity_emb.txt rel_emb_file=rel_emb.txt
    '

