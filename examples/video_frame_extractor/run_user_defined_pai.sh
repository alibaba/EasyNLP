#!/usr/bin/env bash
set -e
# odpscmd to submit odps job
basedir="/home/zhuxiangru.zxr/application/odps_clt_release_64/"
odpscmd="$basedir/bin/odpscmd"

# config file for your odps project
config="$basedir/conf/odps_config_pai_new.ini"

# odps tables for train and dev, format: odps://project_name/tables/table_name
# write_table=odps://pai_exp_dev/tables/clip_wukong_feature

# tables in oss for extraction
# input_table format: id \t video_path
input_table=/data/oss_bucket_0/362425/dataset/video_frame_extraction/examples/annotation_tables/video_id_path_input.txt
video_root_dir=/data/oss_bucket_0/362425/dataset/video_frame_extraction/examples/videos/

# output_table format: id \t [image_path1, ....., image_pathn] or [image_base64_1, ...., image_base64_n]
output_table=/data/oss_bucket_0/362425/dataset/video_frame_extraction/sample_video_frame_output.txt

# tar your package to submit local code to odps
cur_path=/home/zhuxiangru.zxr/workspace/tmp_update/EasyNLP/
cd ${cur_path}
rm -rf entryFile.tar.gz
tar -zcvf entryFile.tar.gz ./examples/video_frame_extractor/ ./requirements.txt


# pai training
echo "starts to land pai job..."
command="
pai -name pytorch180
    -project algo_public
    -Dscript=file://${cur_path}entryFile.tar.gz
    -DentryFile=examples/video_frame_extractor/main.py
    -Dcluster='{\"worker\":{\"cpu\":800}}'
    -Dpython='3.6'
    -DenableDockerFusion=false
    -Dbuckets='oss://easynlp-dev/?role_arn=acs:ram::xxxxxxxxxxxxx:role/easynlp-dev2&host=cn-zhangjiakou.oss.aliyuncs.com'
    -DuserDefinedParameters='\
        --tables=${input_table} \
        --video_root_dir=${video_root_dir} \
        --frame_num=3 \
        --outputs=${output_table}
    '
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
