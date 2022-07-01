export CUDA_VISIBLE_DEVICES=$1

#动态获取GPU数
OLD_IFS="$IFS"
IFS="," 
arr=($1)
IFS="$OLD_IFS"

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

mode=$2

if [ "$mode" = "train" ]; then
  easynlp \
  --mode $mode \
  --worker_gpu=${#arr[@]} \
  --tables=./MUGE_MR_train_base64_part.tsv,./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_model/ \
  --learning_rate=1e-4  \
  --epoch_num=1  \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps 200 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=clip_chinese_roberta_large_with_vit_large fix_vision=True mode=finetune'  
  
elif [ "$mode" = "evaluate" ]; then
  easynlp \
  --mode $mode \
  --worker_gpu=${#arr[@]} \
  --tables=./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_model/ \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip 

elif [ "$mode" = "predict" ]; then
    easynlp \
      --mode $mode \
      --worker_gpu=${#arr[@]} \
      --tables=./MUGE_MR_test_base64_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text_feat \
      --outputs ./text_feat.tsv \
      --first_sequence=text \
      --checkpoint_dir=./clip_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

# elif [ "$mode" = "predict" ]; then
#     easynlp \
#       --mode $mode \
#       --worker_gpu=${#arr[@]} \
#       --tables=./MUGE_MR_test_base64_part_image.tsv \
#       --input_schema=image:str:1 \
#       --output_schema=image_feat \
#       --outputs ./image_feat.tsv \
#       --first_sequence=image \
#       --checkpoint_dir=./clip_model/ \
#       --random_seed=42 \
#       --logging_steps=100 \
#       --save_checkpoint_steps=500 \
#       --sequence_length=32 \
#       --micro_batch_size=2 \
#       --app_name=clip 
fi
