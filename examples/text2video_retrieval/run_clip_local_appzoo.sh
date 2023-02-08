export CUDA_VISIBLE_DEVICES=$1

mode=$2

if [ "$mode" = "train_en" ]; then
  easynlp \
  --mode train \
  --worker_gpu=1 \
  --tables=./msrvtt_subset/MSRVTT_train.tsv,./msrvtt_subset/MSRVTT_test.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip4clip_en_model/ \
  --learning_rate=1e-7  \
  --epoch_num=5  \
  --random_seed=42 \
  --logging_steps=10 \
  --save_checkpoint_steps 1000 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip4clip \
  --save_all_checkpoints \
  --user_defined_parameters='pretrain_model_name_or_path=clip_vit_base_patch32'  

elif [ "$mode" = "evaluate_en" ]; then
  easynlp \
  --mode evaluate \
  --worker_gpu=1 \
  --tables=./msrvtt_subset/MSRVTT_test.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=./clip4clip_en_model/ \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip4clip 

elif [ "$mode" = "predict_en_text" ]; then
    easynlp \
      --mode predict \
      --worker_gpu=1 \
      --tables=./msrvtt_subset/MSRVTT_test_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text_feat \
      --outputs ./msrvtt_subset/MSRVTT_test_text_feat.tsv \
      --first_sequence=text \
      --checkpoint_dir=./clip4clip_en_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=32 \
      --app_name=clip4clip 

elif [ "$mode" = "predict_en_video" ]; then
    easynlp \
      --mode predict \
      --worker_gpu=1 \
      --tables=./msrvtt_subset/MSRVTT_test_part_video.tsv \
      --input_schema=image:str:1 \
      --output_schema=video_feat \
      --outputs ./msrvtt_subset/MSRVTT_test_video_feat.tsv \
      --first_sequence=image \
      --checkpoint_dir=./clip4clip_en_model/ \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=32 \
      --app_name=clip4clip 

fi
