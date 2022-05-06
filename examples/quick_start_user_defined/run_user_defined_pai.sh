# odps tables for train and dev, format: odps://project_name/tables/table_name
#!/usr/bin/env bash
set -e

odpscmd="/Users/zhangtaolin/Desktop/console/bin/odpscmd"
config="/Users/zhangtaolin/Desktop/console/conf/odps_config.ini"

train_table=odps://pai_exp_dev/tables/modelzoo_example_train
dev_table=odps://pai_exp_dev/tables/modelzoo_example_dev

# tar your package to submit local code to odps
cur_path=/Users/zhangtaolin/Desktop/EasyNLP/
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
      --checkpoint_dir=oss://easytransfer-new/311103/test1/ \
      --buckets=\'oss://easytransfer-new/?access_key_id=XXX&access_key_secret=XXX&host=oss-cn-zhangjiakou.aliyuncs.com\'
'
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
