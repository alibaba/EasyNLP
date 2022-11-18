mode=$1

MASTER_ADDR=localhost
MASTER_PORT=6027
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

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

if [ ! -f ./fashiongen_1to1_train.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/fashiongen_1to1_train.tsv
fi

if [ ! -f ./fashiongen_1to1_test.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/fashiongen_1to1_test.tsv
fi

if [ ! -f ./fashiongen_1to1_test_part_image.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/fashiongen_1to1_test_part_image.tsv
fi

if [ ! -f ./fashiongen_1to1_test_part_text.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/fashiongen_1to1_test_part_text.tsv
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ "$mode" = "train_cn" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
  --mode=train \
  --worker_gpu=1 \
  --tables=./MUGE_MR_train_base64_part.tsv,./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_cn_model/ \
  --learning_rate=1e-6  \
  --epoch_num=1  \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps 200 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=alibaba-pai/clip_chinese_roberta_base_vit_base'  
  
elif [ "$mode" = "evaluate_cn" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
  --mode=evaluate \
  --worker_gpu=1 \
  --tables=./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_cn_model \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip 

elif [ "$mode" = "predict_cn_text" ]; then
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
      --mode=predict \
      --worker_gpu=1 \
      --tables=./MUGE_MR_test_base64_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text_feat \
      --outputs ./text_feat.tsv \
      --first_sequence=text \
      --checkpoint_dir=./clip_cn_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

elif [ "$mode" = "predict_cn_image" ]; then
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
      --mode=predict \
      --worker_gpu=1 \
      --tables=./MUGE_MR_test_base64_part_image.tsv \
      --input_schema=image:str:1 \
      --output_schema=image_feat \
      --outputs ./image_feat.tsv \
      --first_sequence=image \
      --checkpoint_dir=./clip_cn_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

elif [ "$mode" = "train_en" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
  --mode=train \
  --worker_gpu=1 \
  --tables=./fashiongen_1to1_train.tsv,./fashiongen_1to1_test.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_en_model/ \
  --learning_rate=1e-6  \
  --epoch_num=1  \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps 200 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=alibaba-pai/pai-clip-commercial-base-en'  

elif [ "$mode" = "evaluate_en" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
  --mode=evaluate \
  --worker_gpu=1 \
  --tables=./fashiongen_1to1_test.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_en_model/ \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=77 \
  --micro_batch_size=32 \
  --app_name=clip 

elif [ "$mode" = "predict_en_text" ]; then
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
      --mode=predict \
      --worker_gpu=1 \
      --tables=./fashiongen_1to1_test_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text_feat \
      --outputs=./fashion_text_feat.tsv \
      --first_sequence=text \
      --checkpoint_dir=./clip_en_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

elif [ "$mode" = "predict_en_image" ]; then
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
      --mode=predict \
      --worker_gpu=1 \
      --tables=./fashiongen_1to1_test_part_image.tsv \
      --input_schema=image:str:1 \
      --output_schema=image_feat \
      --outputs=./fashion_image_feat.tsv \
      --first_sequence=image \
      --checkpoint_dir=./clip_en_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

fi
