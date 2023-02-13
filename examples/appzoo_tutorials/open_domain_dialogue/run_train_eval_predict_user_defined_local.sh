export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/open_domain_dialogue/train.tsv
fi

if [ ! -f ./valid.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/open_domain_dialogue/valid.tsv
fi

if [ ! -f ./persona.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/open_domain_dialogue/persona.tsv
fi

MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode $mode \
    --worker_gpu=1 \
    --tables=train.tsv,valid.tsv \
    --checkpoint_dir=./chat_model/ \
    --learning_rate=3e-5  \
    --epoch_num=3  \
    --random_seed=42 \
    --save_checkpoint_steps=50 \
    --sequence_length=512 \
    --micro_batch_size=1 \
    --app_name=open_domain_dialogue \
    --user_defined_parameters='
        pretrain_model_name_or_path=blender-dialog-90M-en
        label_length=128
    '

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=valid.tsv \
    --micro_batch_size=1 \
    --app_name=open_domain_dialogue \
    --user_defined_parameters='
        pretrain_model_name_or_path=blender-dialog-90M-en
    '


elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=persona.tsv \
    --sequence_length=512 \
    --app_name=open_domain_dialogue \
    --user_defined_parameters='
        pretrain_model_name_or_path=blender-dialog-90M-en
    '

fi
