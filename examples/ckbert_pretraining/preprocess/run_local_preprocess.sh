base_dir=$PWD
save_path=$base_dir/resources
local_original_data=$save_path/original_data_small.json
remote_original_data=https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ckbert/original_data_small.json

if [ ! -f $local_original_data ];then
    wget -P $save_path $remote_original_data
fi

python ltp_extract_relationship.py \
/mnt/djw/latest/EasyNLP/examples/ckbert_pretraining/preprocess/resources/original_data_small.json \
/mnt/djw/latest/EasyNLP/examples/ckbert_pretraining/preprocess/resources/ltp_results.txt \
0,1,2,3

python mask_data.py \
/mnt/djw/latest/EasyNLP/examples/ckbert_pretraining/preprocess/resources/ltp_results.txt \
/mnt/djw/latest/EasyNLP/examples/ckbert_pretraining/preprocess/resources/mask_results.txt \
128 0.3 0.3 0.4