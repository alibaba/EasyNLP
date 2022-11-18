
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

mode=$1

if [ "$mode" = "pretrain_cn" ]; then
if [ ! -f ./COCO_test_images.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/COCO_test_images.tsv
fi

  easynlp \
  --mode train \
  --worker_gpu=1 \
  --tables=local_path_to.tar,./COCO_test_images.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./pretrain_clip_cn_model/ \
  --learning_rate=1e-6  \
  --epoch_num=10  \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps 200 \
  --sequence_length=32 \
  --micro_batch_size=96 \
  --app_name=clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=./pretrain_clip_cn_model/'  

elif [ "$mode" = "train_cn" ]; then
  easynlp \
  --mode train \
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
  --micro_batch_size=2 \
  --app_name=clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=alibaba-pai/clip_chinese_roberta_base_vit_base'  
  
elif [ "$mode" = "evaluate_cn" ]; then
  easynlp \
  --mode evaluate \
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
  --micro_batch_size=2 \
  --app_name=clip 

elif [ "$mode" = "predict_cn_text" ]; then
    easynlp \
      --mode predict \
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
    easynlp \
      --mode predict \
      --worker_gpu=1 \
      --tables=./MUGE_MR_test_base64_part_image.tsv \
      --input_schema=image:str:1 \
      --output_schema=image_feat \
      --outputs ./image_feat.tsv \
      --second_sequence=image \
      --checkpoint_dir=./clip_cn_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

elif [ "$mode" = "train_en" ]; then
  easynlp \
  --mode train \
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
  --micro_batch_size=2 \
  --app_name=clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=alibaba-pai/pai-clip-commercial-base-en'  

elif [ "$mode" = "evaluate_en" ]; then
  easynlp \
  --mode evaluate \
  --worker_gpu=1 \
  --tables=./fashiongen_1to1_test.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip_en_model/ \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=32 \
  --micro_batch_size=2 \
  --app_name=clip 

elif [ "$mode" = "predict_en_text" ]; then
    easynlp \
      --mode predict \
      --worker_gpu=1 \
      --tables=./fashiongen_1to1_test_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text_feat \
      --outputs ./fashion_text_feat.tsv \
      --first_sequence=text \
      --checkpoint_dir=./clip_en_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

elif [ "$mode" = "predict_en_image" ]; then
    easynlp \
      --mode predict \
      --worker_gpu=1 \
      --tables=./fashiongen_1to1_test_part_image.tsv \
      --input_schema=image:str:1 \
      --output_schema=image_feat \
      --outputs ./fashion_image_feat.tsv \
      --second_sequence=image \
      --checkpoint_dir=./clip_en_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=clip 

fi
