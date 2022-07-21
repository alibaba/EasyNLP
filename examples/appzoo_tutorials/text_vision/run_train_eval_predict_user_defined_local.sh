export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./MUGE_MR_train_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_train_base64_part.tsv
fi

if [ ! -f ./MUGE_MR_valid_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_valid_base64_part.tsv
fi

if [ ! -f ./MUGE_MR_test_base64_part_text.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_test_base64_part_text.tsv
fi

if [ ! -f ./MUGE_MR_test_base64_part_image.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_test_base64_part_image.tsv
fi

MASTER_ADDR=localhost
MASTER_PORT=6009
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=MUGE_MR_train_base64_part.tsv,MUGE_MR_valid_base64_part.tsv \
    --input_schema=text:str:1,image:str:1 \
    --first_sequence=text \
    --second_sequence=image \
    --checkpoint_dir=./clip_model/ \
    --learning_rate=1e-4  \
    --epoch_num=1  \
    --random_seed=42 \
    --save_checkpoint_steps=200 \
    --sequence_length=32 \
    --train_batch_size=32 \
    --app_name=clip \
    --user_defined_parameters='
        pretrain_model_name_or_path=clip_chinese_roberta_large_with_vit_large
        fix_vision=True
        mode=finetune
    '

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=MUGE_MR_valid_base64_part.tsv \
    --input_schema=text:str:1,image:str:1 \
    --first_sequence=text \
    --second_sequence=image \
    --checkpoint_dir=./clip_model/ \
    --sequence_length=32 \
    --micro_batch_size=32 \
    --app_name=clip \
    --user_defined_parameters=''

elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=MUGE_MR_valid_base64_part.tsv \
    --outputs=text_feat.tsv \
    --input_schema=text:str:1 \
    --output_schema=text_feat \
    --append_cols=text \
    --first_sequence=text \
    --second_sequence=image \
    --checkpoint_path=./clip_model/ \
    --micro_batch_size=32 \
    --sequence_length=32 \
    --app_name=clip \
    --user_defined_parameters=''

fi

