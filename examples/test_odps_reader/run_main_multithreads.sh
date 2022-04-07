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

# pai training
echo "starts to land pai job..."
command="
pai -name pytorch180
-project algo_public
-Dscript='file:///home/admin/workspace/EasyNLP/examples/test_odps_reader/main_multithreads.py'
-Dtables=${train_table}
-Doutputs=${dev_table}
-DworkerCount=1
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
