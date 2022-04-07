#!/usr/bin/env bash
set -e
# odpscmd to submit odps job
# basedir="/home/admin/workspace/odps_clt_release_64"
basedir="/apsarapangu/disk3/minghui.qmh/odps_clt_release_64"
odpscmd="$basedir/bin/odpscmd"

# config file for your odps project
# config="/apsarapangu/disk3/minghui.qmh/odps_clt_release_64/conf/odps_config.ini"
config="$basedir/conf/odps_config_pai_exp_dev.ini"

# id and secret to access your oss host
id_and_secret_and_host=`cat $basedir/conf/easytransfer-new.id_and_secret_and_path`

# odps tables for train and dev, format: odps://project_name/tables/table_name
train_table=odps://pai_exp_dev/tables/clip_wukong_feature_tmp

# tar your package to submit local code to odps
# cur_path=/home/admin/workspace/EasyNLP/
cur_path=/apsarapangu/disk3/minghui.qmh/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz  ./easynlp/ ./examples/feature_extractor/ ./requirements.txt

# pai training
echo "starts to land pai job..."
command="
pai -name pytorch180
-project algo_public
-Dscript=file://${cur_path}entryFile.tar.gz
-DentryFile=examples/feature_extractor/main_featext.py
-Dcluster='{\"worker\":{\"gpu\":100,\"cpu\":100,\"memory\":10000}}'
-DworkerCount=2
-Dtables=${train_table}
-Dpython='3.6'
-DenableDockerFusion=false
-Dbuckets='oss://easytransfer-new/?role_arn=***&host=***'
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
