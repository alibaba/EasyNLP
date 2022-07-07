export CUDA_VISIBLE_DEVICES=$1

#动态获取GPU数
OLD_IFS="$IFS"
IFS="," 
arr=($1)
IFS="$OLD_IFS"

if [ ! -f ./MUGE_MR_valid_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_valid_base64_part.tsv
fi

easynlp \
  --mode evaluate \
  --worker_gpu=${#arr[@]} \
  --tables=./MUGE_MR_valid_base64_part.tsv \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --checkpoint_dir=wukong_vit_l_14_clip \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=wukong \
  --user_defined_parameters='cosine_similarity=True'

