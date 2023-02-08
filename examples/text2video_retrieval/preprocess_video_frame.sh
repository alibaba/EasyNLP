# Download data
if [ ! -f ./msrvtt_subset/MSRVTT_data.json ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/text2video_retrieval/MSRVTT_subset.zip
    unzip MSRVTT_subset.zip -d msrvtt_subset/
fi

python preprocess_video_frame.py \
    --csv_dir=./msrvtt_subset/MSRVTT_train_subset_100.csv \
    --json_dir=./msrvtt_subset/MSRVTT_data.json \
    --video_dir=./msrvtt_subset/MSRVTT_video_subset \
    --frame_num=12 \
    --frame_dir=./msrvtt_subset/MSRVTT_extracted_frames \
    --output=./msrvtt_subset/MSRVTT_train.tsv

python preprocess_video_frame.py \
    --csv_dir=./msrvtt_subset/MSRVTT_test_subset_100.csv \
    --json_dir=./msrvtt_subset/MSRVTT_data.json \
    --video_dir=./msrvtt_subset/MSRVTT_video_subset \
    --frame_num=12 \
    --frame_dir=./msrvtt_subset/MSRVTT_extracted_frames \
    --output=./msrvtt_subset/MSRVTT_test.tsv
