#!/usr/bin/env bash
set -e
export PATH=$PATH:/Users/tingtingliu/Develops/osscmd_ltt
odpscmd="/Users/tingtingliu/Develops/odpscmd_ltt/bin/odpscmd"
config="/Users/tingtingliu/Develops/odpscmd_ltt/conf/odps_config_pai_exp_dev_tn_hz.ini"
#config="/Users/tingtingliu/Develops/odpscmd_ltt/conf/odps_config_sre_mpi_algo_dev.ini"

ini_path='/Users/tingtingliu/Develops/odpscmd_ltt/conf/pai_exp_dev_zjk_tn.ini'
role_arn_and_host=`cat ${ini_path}`

if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/classification/dev.tsv
fi

access_key_id_path='../../../tools/upload_to_oss_key_id'
access_key_secret_path='../../../tools/upload_to_oss_key_secret'

export input_table=odps://pai_exp_dev/tables/appzoo_example_feature_extraction_input
export output_table=odps://pai_exp_dev/tables/appzoo_example_feature_extraction_output
export model_dir=oss://easytransfer-new/225247/tmp_public/
export oss_bucket_name="easytransfer-new"
export access_key_id=`cat ${access_key_id_path}`
export access_key_secret=`cat ${access_key_secret_path}`
export oss_host=oss-cn-zhangjiakou.aliyuncs.com
export bucket_host=cn-zhangjiakou.oss.aliyuncs.com

command="drop table if exists appzoo_example_feature_extraction_input;"
${odpscmd} --config="${config}" -e "${command}"
command="create table appzoo_example_feature_extraction_input(label STRING, sid1 STRING, sid2 STRING, sent1 STRING,sent2 STRING);"
${odpscmd} --config="${config}" -e "${command}"
command="tunnel upload dev.tsv appzoo_example_feature_extraction_input -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

command="drop table if exists appzoo_example_feature_extraction_output;"
${odpscmd} --config="${config}" -e "${command}"

command="
  pai -name easynlp_dev
  -project algo_platform_dev
  -Dmode=predict
  -DworkerGPU=1
  -DinputTable=${input_table}
  -DoutputTable=${output_table}
  -DinputSchema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1
  -DoutputSchema=pooler_output,first_token_output,all_hidden_outputs
  -DfirstSequence=sent1
  -DcheckpointDir=bert-base-uncased
  -DappendCols=label
  -DbatchSize=32
  -DsequenceLength=128
  -DappName=vectorization
  -DworkerCount=1
  -Dbuckets='oss://${oss_bucket_name}/?role_arn=${role_arn_and_host}'
"
echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."