#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# Local training example
# cur_path=/tmp/EasyNLP
 cur_path=/apsarapangu/disk3/minghui.qmh/EasyNLP/
#cur_path=/wjn/EasyNLP
mode=$2
task=$3

cd ${cur_path}

MASTER_ADDR=localhost
MASTER_PORT=6011
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS benchmarks/clue/main.py \
    --mode=$mode \
    --app_name=text_match \
    --tables=tmp/train.tsv,tmp/dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./tmp/benchmarks/clue/$task \
    --optimizer=BertAdam \
    --learning_rate=5e-5  \
    --epoch_num=10  \
    --random_seed=42 \
    --logging_steps=10 \
    --save_checkpoint_steps=500 \
    --sequence_length=256 \
    --micro_batch_size=32 \
    --user_defined_parameters="clue_name=clue task_name=$task pretrain_model_name_or_path=bert-base-chinese"

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS benchmarks/clue/main_evaluate.py \
    --mode=$mode \
    --worker_gpu=1 \
    --app_name=text_match \
    --tables=tmp/train.tsv,tmp/dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./tmp/benchmarks/clue/$task \
    --sequence_length=256 \
    --micro_batch_size=48 \
    --user_defined_parameters="clue_name=clue task_name=$task"

elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS benchmarks/clue/main_predict.py \
    --mode=$mode \
    --worker_gpu=1 \
    --app_name=text_match \
    --tables=tmp/train.tsv,tmp/dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./tmp/benchmarks/clue/$task \
    --sequence_length=256 \
    --micro_batch_size=48 \
    --user_defined_parameters="clue_name=clue task_name=$task"

fi