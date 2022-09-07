# odps tables for train and dev, format: odps://project_name/tables/table_name
#!/usr/bin/env bash
set -e

odpscmd="/home/zhuxiangru.zxr/application/odps_clt_release_64/bin/odpscmd"
config="/home/zhuxiangru.zxr/application/odps_clt_release_64/conf/odps_config_pai_new.ini"

# input and output path
train_table=/data/oss_bucket_0/362425/tmp_test_pai/video2text_generation/datasets/VC_train_path.txt
dev_table=/data/oss_bucket_0/362425/tmp_test_pai/video2text_generation/datasets/VC_val_path.txt
image_root_dir=/data/oss_bucket_0/362425/tmp_test_pai/video2text_generation/datasets/sample_frame_images/
checkpoint_dir=/data/oss_bucket_0/362425/tmp_test_pai/video2text_generation/vit_gpt_pretrain_model/

# tar your package to submit local code to odps
cur_path=$PWD/../../
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz  ./easynlp/ ./examples/video2text_generation/ ./requirements.txt

user_defined_parameters='enable_img_path=True \
      img_root_dir='${image_root_dir}' \
      frame_num=4 \
      vit_ckpt_path=ViT-L/14 \
      img_size=224 \
      img_len=256 \
      text_len=32 \
      pretrain_model_name_or_path=bert-base-chinese \
      block_size=288 \
      n_layer=12 \
      n_head=12 \
      n_embd=768'

# base: block_size=288 n_layer=12 n_head=12 n_embd=768
# large: block_size=288 n_layer=24 n_head=16 n_embd=1024


command="
pai -name pytorch180
-project algo_public
-Dscript=file://${cur_path}entryFile.tar.gz
-DentryFile=examples/video2text_generation/main.py
-Dcluster='{\"worker\":{\"gpu\":800,\"cpu\":800,\"memory\":160000}}'
-Dpython='3.6'
-DenableDockerFusion=false
-Dbuckets='oss://easynlp-dev/?role_arn=acs:ram::xxxxxxxxxxx:role/easynlp-dev2&host=cn-zhangjiakou.oss.aliyuncs.com'
-DuserDefinedParameters='--mode=train \
      --worker_gpu=1 \
      --tables=${train_table},${dev_table} \
      --input_schema=id:str:1,imgpath:str:1,text:str:1 \
      --first_sequence=imgpath \
      --second_sequence=text \
      --checkpoint_dir=${checkpoint_dir} \
      --learning_rate=4e-5 \
      --epoch_num=30  \
      --random_seed=42 \
      --logging_steps=10 \
      --save_checkpoint_steps=50 \
      --sequence_length=288 \
      --micro_batch_size=24 \
      --app_name=video2text_generation \
      --user_defined_parameters=\'${user_defined_parameters}\'
   '
"



echo "${command}"
${odpscmd} --config="${config}" -e "${command}"



