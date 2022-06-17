#!/bin/bash
set -e
# prepare arguments
nproc_per_node=8
nnodes=1
node_rank=0
addr="127.0.0.1"
port="9002"
use_torchx=1

EPOCH=5
LOGGING_STEPS=10
SAVING_CHECHPONINT_STEPS=2150
SEQUENCE_LEN=128
TRAIN_BATCH_SIZE=48
# oss bucket
HOME_DIR=/data/oss_bucket_0/373535
DATASET=$HOME_DIR/easynlp_pretrain/datasets/zh_train20Ths.json
MODEL_NAME=hfl/macbert-base-zh
# checkpoint save path
CHECKPOINT_DIR=$HOME_DIR/easynlp_pretrain/pretrain_results/test

# pai config

basedir="/apsarapangu/disk3/hexi.ltt/odps_clt_release_64"  # 136服务器
odpscmd="$basedir/bin/odpscmd"
# config="$basedir/conf/odps_config_sre_new.ini"   # 可用于finetune
config="$basedir/conf/odps_config_pai_new.ini"     # 可用于pretrain


cur_path=/home/dongjunwei.djw/workspace/easynlp_pretrain/EasyNLP
cd ${cur_path}
rm -f proj.tar.gz
tar -zcf proj.tar.gz *
job_path='file://'${cur_path}'/proj.tar.gz'

task_name=pytorch180
command="
pai -name ${task_name}
-project algo_public
-Dscript=${job_path}
-DentryFile='-m easypai.torch.launch --not_assign_devices ./examples/appzoo_tutorials/language_modeling/main.py'
-Dpython='3.6'
-Dbuckets='xxxxxxxxxxx'
-DenableDockerFusion=false
-Doversubscription=false
-Dcluster='{\"worker\":{\"gpu\":800,\"cpu\":900,\"memory\":200000}}'
-DworkerCount=1
-DuserDefinedParameters='\
      --mode=train \
      --worker_gpu=8 \
      --tables=\"$DATASET,\" \
      --learning_rate=5e-5  \
      --epoch_num=$EPOCH  \
      --logging_steps=$LOGGING_STEPS \
      --save_checkpoint_steps=$SAVING_CHECHPONINT_STEPS \
      --sequence_length=$SEQUENCE_LEN \
      --train_batch_size=$TRAIN_BATCH_SIZE \
      --checkpoint_dir=$CHECKPOINT_DIR \
      --app_name=language_modeling \
      --use_amp \
      --save_all_checkpoints \
      --user_defined_parameters=\"pretrain_model_name_or_path=$MODEL_NAME\"
'
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
