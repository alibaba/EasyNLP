#!/usr/bin/env bash
set -e
# odpscmd to submit odps job
basedir="/home/wangxiaodan.wxd/workspace/odps_clt_release_64"
odpscmd="$basedir/bin/odpscmd"

# config file for your odps project
config="$basedir/conf/odps_config.ini"

# id and secret to access your oss host
id_and_secret_and_host=`cat $basedir/conf/easytransfer-new.id_and_secret_and_path`

# tar your package to submit local code to odps
cur_path=/home/wangxiaodan.wxd/workspace/Dev/commit/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz  ./easynlp/ ./examples/blip_retrieval/ ./requirements.txt

# pai training
echo "starts to land pai job..."
command="
pai -name pytorch180
-project algo_public
-Dscript=file://${cur_path}entryFile.tar.gz
-DentryFile=examples/blip_retrieval/train_retrieval.py
-Dcluster='{\"worker\":{\"gpu\":400,\"cpu\":100,\"memory\":10000}}'
-DworkerCount=1
-Dpython='3.6'
-Dbuckets='oss://easytransfer-new/?role_arn=***&host=***'
-DenableDockerFusion=false
-DuserDefinedParameters='--output_dir \"/data/oss_bucket_0/363811/models/blip/output/finetune_flickr\" \
      --evaluate \
      --image_root=\"/data/oss_bucket_0/363811/text_image/flickr30k/\" \
      --ann_root=\"/data/oss_bucket_0/363811/text_image/flickr30k/annotation/\" \
      --dataset=\"flickr\" \
      --train_file=\"flickr30k_train.json\" \
      --val_file=\"flickr30k_val.json\" \
      --test_file=\"flickr30k_test.json\" \
      --pretrained=\"http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/blip/pretrained/model_base_retrieval_flickr.pth\" \
      --vit=\"base\" \
      --batch_size_train=32 \
      --batch_size_test=64 \
      --vit_grad_ckpt=False \
      --vit_ckpt_layer=4 \
      --init_lr=1e-5 \
      --image_size=384 \
      --queue_size=57600 \
      --alpha=0.4 \
      --k_test=128 \
      --negative_all_rank=False \
      --weight_decay=0.05 \
      --min_lr=0 \
      --max_epoch=6'
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
