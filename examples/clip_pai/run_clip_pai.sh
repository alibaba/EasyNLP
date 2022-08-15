#!/usr/bin/env bash
set -e

basedir="/apsarapangu/disk3/hexi.ltt/odps_clt_release_64"
odpscmd="$basedir/bin/odpscmd"
config="$basedir/conf/odps_config_pai_new.ini"

if [ ! -f ./MUGE_MR_train_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_train_base64_part.tsv
fi

if [ ! -f ./MUGE_MR_valid_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_valid_base64_part.tsv
fi

command="drop table if exists modelzoo_example_clip_train;"
${odpscmd} --config="${config}" -e "${command}"
command="create table modelzoo_example_clip_train(text STRING, image STRING);"
${odpscmd} --config="${config}" -e "${command}"
command="tunnel upload MUGE_MR_train_base64_part.tsv modelzoo_example_clip_train -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

command="drop table if exists modelzoo_example_clip_valid;"
${odpscmd} --config="${config}" -e "${command}"
command="create table modelzoo_example_clip_valid(text STRING, image STRING);"
${odpscmd} --config="${config}" -e "${command}"
command="tunnel upload MUGE_MR_valid_base64_part.tsv modelzoo_example_clip_valid -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

export train_table=odps://pai_exp_dev/tables/modelzoo_example_clip_train
export dev_table=odps://pai_exp_dev/tables/modelzoo_example_clip_valid
export model_dir=oss://easynlp-dev/317042/temp/

# tar your package to submit local code to odps
cur_path=/apsarapangu/disk2/zhangjie.ll/pai_clip/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz  ./easynlp/ ./examples/clip_pai/ ./requirements.txt

command="
pai -name pytorch180
-project=algo_public
-Dscript=file://${cur_path}entryFile.tar.gz
-DentryFile=examples/clip_pai/main.py
-Dcluster='{\"worker\":{\"gpu\":100,\"cpu\":100,\"memory\":10000}}'
-Dtables=${train_table},${dev_table}
-Dpython='3.6'
-DenableDockerFusion=false
-DuserDefinedParameters='--mode=train \
  --worker_gpu=1 \
  --input_schema=text:str:1,image:str:1 \
  --first_sequence=text \
  --second_sequence=image \
  --user_defined_parameters=\'pretrain_model_name_or_path=clip_chinese_roberta_large_with_vit_large\' \
  --learning_rate=1e-4  \
  --random_seed=42 \
  --epoch_num=1  \
  --logging_steps=100 \
  --save_checkpoint_steps=200 \
  --sequence_length=32 \
  --micro_batch_size=32 \
  --app_name=clip \
  --checkpoint_dir=oss://easytransfer-new/311103/test1/ \
  --buckets=\'oss://xxxxxxxxxxx\' 
  '"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
