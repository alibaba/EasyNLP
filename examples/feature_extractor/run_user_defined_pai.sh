#!/usr/bin/env bash
set -e
# odpscmd to submit odps job
# basedir="/apsarapangu/disk3/hexi.ltt/odps_clt_release_64"
basedir="/home/wangxiaodan.wxd/workspace/odps_clt_release_64"
odpscmd="$basedir/bin/odpscmd"

# config file for your odps project
# config="/apsarapangu/disk3/minghui.qmh/odps_clt_release_64/conf/odps_config.ini"
config="$basedir/conf/odps_config.ini"

# id and secret to access your oss host
id_and_secret_and_host=`cat $basedir/conf/easytransfer-new.id_and_secret_and_path`

# odps tables for train and dev, format: odps://project_name/tables/table_name
write_table=odps://pai_exp_dev/tables/clip_wukong_feature
# dev_table=odps://sre_mpi_algo_dev/tables/modelzoo_example_dev

# tar your package to submit local code to odps
cur_path=/home/wangxiaodan.wxd/workspace/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz ./examples/feature_extractor/ ./requirements.txt

# pai training
echo "starts to land pai job..."
command="
pai -name pytorch180
    -Dscript=file://${cur_path}entryFile.tar.gz
    -DentryFile=examples/feature_extractor/pai_feature_extractor.py
    -Dcluster='{\"worker\":{\"gpu\":100,\"cpu\":800,\"memory\":400000}}'
 	-Doutputs=${write_table}
    -DworkerCount=4;
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
