#!/usr/bin/env bash
set -e
export PATH=$PATH:/Users/jerry/Develops/osscmd
odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
#config="/Users/jerry/Develops/odpscmd/conf/odps_config_sre_mpi_algo_dev.ini"

ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tn.ini'
role_arn_and_host=`cat ${ini_path}`

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/tutorials/ez_text_match/afqmc_public/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/tutorials/ez_text_match/afqmc_public/dev.csv
fi

access_key_id_path='../../../../tools/upload_to_oss_key_id'
access_key_secret_path='../../../../tools/upload_to_oss_key_secret'

export train_table=odps://pai_exp_dev/tables/text_match_two_tower_example_train
export dev_table=odps://pai_exp_dev/tables/text_match_two_tower_example_dev
export dev_table_output=odps://pai_exp_dev/tables/text_match_two_tower_example_dev_output
export model_dir=oss://easytransfer-new/225247/tmp_public/
export oss_bucket_name="easytransfer-new"
export access_key_id=`cat ${access_key_id_path}`
export access_key_secret=`cat ${access_key_secret_path}`
export oss_host=oss-cn-zhangjiakou.aliyuncs.com
export bucket_host=cn-zhangjiakou.oss.aliyuncs.com

if [ "$1" = "train" ]; then

  command="drop table if exists text_match_two_tower_example_train;"
  ${odpscmd} --config="${config}" -e "${command}"
  command="create table text_match_two_tower_example_train(example_id STRING, sent1 STRING, sent2 STRING, label STRING,cate STRING, score STRING);"
  ${odpscmd} --config="${config}" -e "${command}"
  command="tunnel upload train.csv text_match_two_tower_example_train -fd '\t';"
  ${odpscmd} --config="${config}" -e "${command}"

  command="drop table if exists text_match_two_tower_example_dev;"
  ${odpscmd} --config="${config}" -e "${command}"
  command="create table text_match_two_tower_example_dev(example_id STRING, sent1 STRING, sent2 STRING, label STRING,cate STRING, score STRING);"
  ${odpscmd} --config="${config}" -e "${command}"
  command="tunnel upload dev.csv text_match_two_tower_example_dev -fd '\t';"
  ${odpscmd} --config="${config}" -e "${command}"

  command="
  pai -name easynlp_dev
  -project algo_platform_dev
  -Dmode=train
  -DinputTable=${train_table},${dev_table}
  -DinputSchema=example_id:str:1,sent1:str:1,sent2:str:1,label:str:1,cate:str:1,score:str:1
  -DcheckpointDir=${model_dir}tmp_easynlp_text_match_two_tower_examples
  -DfirstSequence=sent1
  -DsecondSequence=sent2
  -DlabelName=label
  -DlabelEnumerateValues=0,1
  -DsequenceLength=64
  -DappName=text_match
  -DlearningRate=3e-5
  -DnumEpochs=3
  -DsaveCheckpointSteps=50
  -DbatchSize=32
  -DworkerCount=1
  -DworkerGPU=1
  -DuserDefinedParameters='
      pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
      two_tower=True
      loss_type=hinge_loss
      margin=0.45
      gamma=32
  '
  -Dbuckets='oss://${oss_bucket_name}?access_key_id=${access_key_id}&access_key_secret=${access_key_secret}&host=${bucket_host}'
  "

fi

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
