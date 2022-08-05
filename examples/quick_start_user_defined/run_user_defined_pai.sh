# odps tables for train and dev, format: odps://project_name/tables/table_name
#!/usr/bin/env bash
set -e

odpscmd="/apsarapangu/disk2/zhuxiangru.zxr/application/odps_clt_release_64/bin/odpscmd"
config="/apsarapangu/disk2/zhuxiangru.zxr/application/odps_clt_release_64/conf/odps_config_pai_new.ini"

train_table=odps://pai_exp_dev/tables/modelzoo_example_train
dev_table=odps://pai_exp_dev/tables/modelzoo_example_dev

# tar your package to submit local code to odps
cur_path=/home/zhuxiangru.zxr/workspace/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz  ./easynlp/ ./examples/quick_start_user_defined/ ./requirements.txt

command="
pai -name pytorch180
-project algo_public
-Dscript=file://${cur_path}entryFile.tar.gz
-DentryFile=examples/quick_start_user_defined/main.py
-Dcluster='{\"worker\":{\"gpu\":100,\"cpu\":100,\"memory\":10000}}'
-Dtables=${train_table},${dev_table}
-Dpython='3.6'
-DenableDockerFusion=false
-Dbuckets='oss://easynlp-dev/?role_arn=xxxxxxx&host=cn-zhangjiakou.oss.aliyuncs.com'
-DuserDefinedParameters='--mode=train \
      --worker_gpu=1 \
      --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
      --first_sequence=sent1 \
      --second_sequence=sent2 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --user_defined_parameters=\'pretrain_model_name_or_path=bert-small-uncased\' \
      --learning_rate=3e-5 \
      --random_seed=42 \
      --epoch_num=1  \
      --logging_steps=1 \
      --save_checkpoint_steps=50 \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=text_classify \
      --checkpoint_dir=/data/oss_bucket_0/362425/tmp_test_user_defined_pai/ 
      '
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"



