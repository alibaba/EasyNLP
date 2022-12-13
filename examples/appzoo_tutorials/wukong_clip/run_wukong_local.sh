mode=$1

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


if [ "$mode" = "train" ]; then
  easynlp \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./MUGE_MR_train_base64_part.tsv,./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./wukong_finetuned/ \
  --learning_rate=1e-4  \
  --epoch_num=1  \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps 200 \
  --sequence_length=32 \
  --micro_batch_size=2 \
  --app_name=wukong_clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=wukong_vit_l_14_clip'  
  
elif [ "$mode" = "evaluate" ]; then
  easynlp \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./wukong_finetuned/ \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=32 \
  --micro_batch_size=2 \
  --app_name=wukong_clip 

elif [ "$mode" = "predict_text" ]; then
    easynlp \
      --mode=predict \
      --worker_gpu=1 \
      --tables=./MUGE_MR_test_base64_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text_feat \
      --outputs=./wukong_text_feat.tsv \
      --first_sequence=text \
      --checkpoint_dir=./wukong_finetuned/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=wukong_clip 

elif [ "$mode" = "predict_image" ]; then

  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./MUGE_MR_test_base64_part_image.tsv \
    --input_schema=image:str:1 \
    --output_schema=image_feat \
    --outputs=./wukong_image_feat.tsv \
    --second_sequence=image \
    --checkpoint_dir=./wukong_finetuned/ \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=500 \
    --sequence_length=32 \
    --micro_batch_size=2 \
    --app_name=wukong_clip
fi
