#!/usr/bin/env bash

set -e
odpscmd="/Users/minghui/Desktop/Projects/odps_clt_release_64/bin/odpscmd"
id_and_secrect=`cat /Users/minghui/Desktop/Projects/odps_clt_release_64/conf/config.id_and_secrect` 
export model_dir=oss://easytransfer-new/104239
export bucket_host=cn-zhangjiakou.oss.aliyuncs.com

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/train.csv
  command1="drop table if exists sequence_labeling_train;"
  command2="create table sequence_labeling_train(content STRING, label STRING);"
  command3="tunnel upload train.csv sequence_labeling_train -fd '\t';"
  ${odpscmd} -e "${command1}"
  ${odpscmd} -e "${command2}"
  ${odpscmd} -e "${command3}"
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/dev.csv
  command1="drop table if exists sequence_labeling_dev;"
  command2="create table sequence_labeling_dev(content STRING, label STRING);"
  command3="tunnel upload dev.csv sequence_labeling_dev -fd '\t';"
  ${odpscmd} -e "${command1}"
  ${odpscmd} -e "${command2}"
  ${odpscmd} -e "${command3}"
fi

echo 'Tar pkg and upload to oss dir...'
tar -zcvf entryFile.tar.gz main.py
~/Desktop/Projects/OSS_Python_API_20140509/ossutilmac64 cp -f entryFile.tar.gz ${model_dir}/
rm -f entryFile.tar.gz

command="
pai -name easynlp_dev
-project algo_platform_dev
-Dscript=${model_dir}/entryFile.tar.gz
-DentryFile=main.py
-DinputTable=odps://sre_mpi_algo_dev/tables/sequence_labeling_train,odps://sre_mpi_algo_dev/tables/sequence_labeling_dev
-DuserDefinedParameters='
      --mode=train
      --checkpoint_dir=${model_dir}/seq_labeling/
      --input_schema=content:str:1,label:str:1
      --first_sequence=content \
      --label_name=label \
      --label_enumerate_values=O,B-ORG,I-ORG,B-LOC,I-LOC,B-PER,I-PER \
      --pretrained_model_name_or_path=hfl/chinese-roberta-wwm-ext \
      --learning_rate=3e-5  \
      --seed=42 \
      --epoch_num=3  \
      --logging_steps=100 \
      --save_checkpoint_steps=100 \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=sequence_labeling'
-Dbuckets=oss://easytransfer-new/?${id_and_secrect}&host=${bucket_host}
-DworkerCount=1
-DworkerGPU=1
"

echo "${command}"
${odpscmd} -e "${command}"
echo "finish..."