# Download data
if [ ! -f ./msrvtt_data/MSRVTT_data.json ]; then
    wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
    wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
    unzip msrvtt_data.zip
    unzip MSRVTT.zip -d msrvtt_data/
    rm msrvtt_data.zip
    rm MSRVTT.zip
fi

python preprocess_video_frame.py \
    --csv_dir=./msrvtt_data/MSRVTT_train.9k.csv \
    --json_dir=./msrvtt_data/MSRVTT_data.json \
    --video_dir=./msrvtt_data/MSRVTT/videos/all \
    --frame_num=12 \
    --frame_dir=./msrvtt_data/MSRVTT_extracted_frames \
    --output=./msrvtt_data/MSRVTT_train.tsv

python preprocess_video_frame.py \
    --csv_dir=./msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --json_dir=./msrvtt_data/MSRVTT_data.json \
    --video_dir=./msrvtt_data/MSRVTT/videos/all \
    --frame_num=12 \
    --frame_dir=./msrvtt_data/MSRVTT_extracted_frames \
    --output=./msrvtt_data/MSRVTT_test_all.tsv
