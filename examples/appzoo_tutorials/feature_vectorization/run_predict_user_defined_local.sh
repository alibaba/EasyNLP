export CUDA_VISIBLE_DEVICES=$1


if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

MASTER_ADDR=localhost
MASTER_PORT=2345
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "predict" ]; then

    python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=dev.tsv \
    --outputs=dev.pred.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --output_schema=pooler_output,first_token_output,all_hidden_outputs \
    --first_sequence=sent1 \
    --checkpoint_dir=./classification_model/ \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --app_name=vectorization

fi
