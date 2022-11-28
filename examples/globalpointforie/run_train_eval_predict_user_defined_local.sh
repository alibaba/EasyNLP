export CUDA_VISIBLE_DEVICES=$1

MASTER_ADDR=localhost
MASTER_PORT=6009
GPUS_PER_NODE=7
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode $mode \
    --learning_rate=2e-05  \
    --fp16 \
    --worker_gpu=7 \
    --app_name=globalpointforie \
    --random_seed=42 \
    --save_checkpoint_steps=500 \
    --sequence_length=512 \
    --weight_decay=0.0 \
    --micro_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --epoch_num=3  \
    --checkpoint_dir=./globalpointforie_model/ \
    --data_threads=5 \
    --user_defined_parameters='pretrain_model_name_or_path=hfl/macbert-large-zh data_dir=./data'

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode $mode \
    --fp16 \
    --data_dir=./data \
    --worker_gpu=7 \
    --app_name=globalpointforie \
    --sequence_length=512 \
    --weight_decay=0.0 \
    --micro_batch_size=4 \
    --checkpoint_dir=./globalpointforie_model/ \
    --data_threads=5 \
    --user_defined_parameters='data_dir=./data'

elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --fp16 \
    --data_dir=./data \
    --worker_gpu=7 \
    --app_name=globalpointforie \
    --sequence_length=512 \
    --weight_decay=0.0 \
    --micro_batch_size=4 \
    --checkpoint_dir=./globalpointforie_model/ \
    --data_threads=5 \
    --user_defined_parameters='data_dir=./data'
fi