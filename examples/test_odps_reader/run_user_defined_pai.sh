#!/usr/bin/env bash
set -e
# odpscmd to submit odps job
basedir="/home/admin/workspace/odps_clt_release_64"
odpscmd="$basedir/bin/odpscmd"

# config file for your odps project
# config="/apsarapangu/disk3/minghui.qmh/odps_clt_release_64/conf/odps_config.ini"
config="$basedir/conf/odps_config_pai_exp_dev.ini"

# id and secret to access your oss host
id_and_secret_and_host=`cat $basedir/conf/easytransfer-new.id_and_secret_and_path`

# odps tables for train and dev, format: odps://project_name/tables/table_name
train_table=odps://sre_mpi_algo_dev/tables/modelzoo_example_train
dev_table=odps://sre_mpi_algo_dev/tables/modelzoo_example_dev

# tar your package to submit local code to odps
cur_path=/home/admin/workspace/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz  ./easynlp/ ./examples/test_odps_reader/ ./requirements.txt

# pai training
echo "starts to land pai job..."
command="
pai -name pytorch180
-project algo_public
-Dscript=file://${cur_path}entryFile.tar.gz
-DentryFile=examples/test_odps_reader/main.py
-Dcluster='{\"worker\":{\"gpu\":100,\"cpu\":100,\"memory\":10000}}'
-DworkerCount=1
-Dtables=${train_table},${dev_table}
-Dpython='3.6'
-DenableDockerFusion=false
-DuserDefinedParameters='--mode=train \
      --worker_gpu=1 \
      --data_threads=2 \
      --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
      --first_sequence=sent1 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --user_defined_parameters=\' pretrain_model_name_or_path=bert-small-uncased\' \
      --learning_rate=3e-5 \
      --random_seed=42 \
      --epoch_num=3  \
      --logging_steps=100 \
      --save_checkpoint_steps=50 \
      --sequence_length=128 \
      --micro_batch_size=3 \
      --app_name=text_classify \
      --checkpoint_dir=oss://easytransfer-new/104239/tmp_examples_2 \
      --buckets=\'oss://easytransfer-new?${id_and_secret_and_host}\'
'
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
