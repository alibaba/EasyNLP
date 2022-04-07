#!/usr/bin/env bash

set -e
odpscmd="/Users/minghui/Desktop/Projects/odps_clt_release_64/bin/odpscmd"
id_and_secrect=`cat /Users/minghui/Desktop/Projects/odps_clt_release_64/conf/config.id_and_secrect` 
export model_dir=oss://easytransfer-new/104239
export bucket_host=cn-zhangjiakou.oss.aliyuncs.com

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/sequence_labeling/train.csv
  command1="drop table if exists sequence_labeling_train;"
  command2="create table sequence_labeling_train(content STRING, label STRING);"
  command3="tunnel upload train.csv sequence_labeling_train -fd '\t';"
  ${odpscmd} -e "${command1}"
  ${odpscmd} -e "${command2}"
  ${odpscmd} -e "${command3}"
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/sequence_labeling/dev.csv
  command1="drop table if exists sequence_labeling_dev;"
  command2="create table sequence_labeling_dev(content STRING, label STRING);"
  command3="tunnel upload dev.csv sequence_labeling_dev -fd '\t';"
  ${odpscmd} -e "${command1}"
  ${odpscmd} -e "${command2}"
  ${odpscmd} -e "${command3}"
fi

command="
pai -name easynlp_dev
-project algo_platform_dev
-Dmode=train
-DinputTable=odps://sre_mpi_algo_dev/tables/sequence_labeling_train,odps://sre_mpi_algo_dev/tables/sequence_labeling_dev
-DinputSchema=content:str:1,label:str:1
-DcheckpointDir=${model_dir}/seq_labeling/
-DfirstSequence=content
-DlabelName=label
-DlabelEnumerateValues=O,B-ORG,I-ORG,B-LOC,I-LOC,B-PER,I-PER
-DsequenceLength=128
-DappName=sequence_labeling
-DlearningRate=3e-5
-DnumEpochs=3
-DsaveCheckpointSteps=100
-DbatchSize=32
-DworkerCount=1
-DworkerGPU=2
-DpretrainedModelNameOrPath=hfl/chinese-roberta-wwm-ext
-Dbuckets=oss://easytransfer-new/?${id_and_secrect}&host=${bucket_host}
"

echo "${command}"
${odpscmd} -e "${command}"
echo "finish..."
