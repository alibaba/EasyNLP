#!/usr/bin/env bash
set -e
export PATH=$PATH:/Users/jerry/Develops/osscmd
odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
#config="/Users/jerry/Develops/odpscmd/conf/odps_config_sre_mpi_algo_dev.ini"

ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tn.ini'
role_arn_and_host=`cat ${ini_path}`

if [ ! -f ./train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/classification/train.tsv
fi

if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/classification/dev.tsv
fi

access_key_id_path='../../../../tools/upload_to_oss_key_id'
access_key_secret_path='../../../../tools/upload_to_oss_key_secret'

export train_table=odps://pai_exp_dev/tables/modelzoo_example_train
export dev_table=odps://pai_exp_dev/tables/modelzoo_example_dev
export dev_table_output=odps://pai_exp_dev/tables/modelzoo_example_dev_output
export model_dir=oss://easytransfer-new/225247/tmp_public/
export oss_bucket_name="easytransfer-new"
export access_key_id=`cat ${access_key_id_path}`
export access_key_secret=`cat ${access_key_secret_path}`
export oss_host=oss-cn-zhangjiakou.aliyuncs.com
export bucket_host=cn-zhangjiakou.oss.aliyuncs.com

if [ "$1" = "train" ]; then

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

  command="
  pai -name easynlp_dev
  -project algo_platform_dev
  -Dmode=train
  -DinputTable=${train_table},${dev_table}
  -DinputSchema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1
  -DcheckpointDir=${model_dir}tmp_easynlp_modelzoo_examples
  -DfirstSequence=sent1
  -DsecondSequence=sent2
  -DlabelName=label
  -DlabelEnumerateValues=0,1
  -DsequenceLength=64
  -DappName=text_classify
  -DlearningRate=3e-5
  -DnumEpochs=1
  -DsaveCheckpointSteps=50
  -DbatchSize=32
  -DworkerCount=1
  -DworkerGPU=1
  -DuserDefinedParameters='
      pretrain_model_name_or_path=bert-base-uncased
  '
  -Dbuckets='oss://${oss_bucket_name}?access_key_id=${access_key_id}&access_key_secret=${access_key_secret}&host=${bucket_host}'
  "

elif [ "$1" = "predict" ]; then

  command="drop table if exists modelzoo_example_dev;"
  ${odpscmd} --config="${config}" -e "${command}"
  command="create table modelzoo_example_dev(label STRING, sid1 STRING, sid2 STRING, sent1 STRING,sent2 STRING);"
  ${odpscmd} --config="${config}" -e "${command}"
  command="tunnel upload dev.tsv modelzoo_example_dev -fd '\t';"
  ${odpscmd} --config="${config}" -e "${command}"

  command="drop table if exists modelzoo_example_dev_output;"
  ${odpscmd} --config="${config}" -e "${command}"

  command="
  pai -name easynlp_dev
  -project algo_platform_dev
  -Dmode=predict
  -DinputTable=${dev_table}
  -DinputSchema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1
  -DoutputTable=${dev_table_output}
  -DoutputSchema=predictions,probabilities,logits,output
  -DcheckpointDir=${model_dir}tmp_easynlp_modelzoo_examples
  -DappendCols=label
  -DfirstSequence=sent1
  -DsecondSequence=sent2
  -DsequenceLength=64
  -DappName=text_classify
  -DbatchSize=4
  -DworkerCount=1
  -DworkerCPU=1
  -DworkerGPU=1
  -Denable_elastic_inference=true
  -Duse_oversold_res=true
  -Dbuckets='oss://${oss_bucket_name}/?${role_arn_and_host}'
  "

fi

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
