#!/usr/bin/env bash
set -e
export PATH=$PATH:/Users/jerry/Develops/osscmd
odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
#config="/Users/jerry/Develops/odpscmd/conf/odps_config_sre_mpi_algo_dev.ini"

if [ ! -f ./train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
fi

if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

command="drop table if exists modelzoo_example_train;"
${odpscmd} --config="${config}" -e "${command}"
command="create table modelzoo_example_train(label STRING, sid1 STRING, sid2 STRING, sent1 STRING,sent2 STRING);"
${odpscmd} --config="${config}" -e "${command}"
command="tunnel upload train.tsv modelzoo_example_train -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

command="drop table if exists modelzoo_example_dev;"
${odpscmd} --config="${config}" -e "${command}"
command="create table modelzoo_example_dev(label STRING, sid1 STRING, sid2 STRING, sent1 STRING,sent2 STRING);"
${odpscmd} --config="${config}" -e "${command}"
command="tunnel upload dev.tsv modelzoo_example_dev -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"


access_key_id_path='../../../../tools/upload_to_oss_key_id'
access_key_secret_path='../../../../tools/upload_to_oss_key_secret'

export train_table=odps://pai_exp_dev/tables/modelzoo_example_train
export dev_table=odps://pai_exp_dev/tables/modelzoo_example_dev
export model_dir=oss://easytransfer-new/225247/tmp_public/
export oss_bucket_name="easytransfer-new"
export access_key_id=`cat ${access_key_id_path}`
export access_key_secret=`cat ${access_key_secret_path}`
export oss_host=oss-cn-zhangjiakou.aliyuncs.com
export bucket_host=cn-zhangjiakou.oss.aliyuncs.com

tar -zcvf entryFile.tar.gz main.py
ossutilmac64 cp -f entryFile.tar.gz ${model_dir} -i ${access_key_id} -k ${access_key_secret} -e ${oss_host}
rm -f entryFile.tar.gz

command="
pai -name easynlp_dev
-project algo_platform_dev
-Dscript=oss://easytransfer-new/225247/tmp_public/entryFile.tar.gz
-DentryFile=main.py
-DinputTable=${train_table},${dev_table}
-DuserDefinedParameters='
      --mode=train
      --checkpoint_dir=${model_dir}tmp_easynlp_modelzoo_examples/
      --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1
      --first_sequence=sent1 \
      --second_sequence=sent2 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --pretrained_model_name_or_path=bert-base-uncased \
      --learning_rate=3e-5  \
      --random_seed=42 \
      --epoch_num=3  \
      --logging_steps=100 \
      --save_checkpoint_steps=50 \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=text_classify
'
-Dbuckets='oss://${oss_bucket_name}?access_key_id=${access_key_id}&access_key_secret=${access_key_secret}&host=${bucket_host}'
-DworkerCount=1
-DworkerGPU=1
"
echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
