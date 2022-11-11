export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/dev.csv
fi

MASTER_ADDR=localhost
MASTER_PORT=6008
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_labeling/main.py \
    --mode $mode \
    --tables=./train.csv,./dev.csv \
    --input_schema=content:str:1,label:str:1 \
    --first_sequence=content \
    --label_name=label \
    --label_enumerate_values=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
    --checkpoint_dir=./seq_labeling/ \
    --learning_rate=3e-5  \
    --epoch_num=1  \
    --save_checkpoint_steps=50 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --app_name=sequence_labeling \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
    '

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --tables=dev.csv \
    --input_schema=content:str:1,label:str:1 \
    --first_sequence=content \
    --label_name=label \
    --label_enumerate_values=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
    --checkpoint_dir=./seq_labeling/ \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --app_name=sequence_labeling \


elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --tables=dev.csv \
    --outputs=dev.pred.csv \
    --input_schema=content:str:1,label:str:1 \
    --output_schema=output \
    --append_cols=label \
    --first_sequence=content \
    --checkpoint_path=./seq_labeling/ \
    --micro_batch_size 32 \
    --sequence_length=128 \
    --app_name=sequence_labeling \

fi