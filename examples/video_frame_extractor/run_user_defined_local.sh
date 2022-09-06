#! pip install decord  #https://github.com/dmlc/decord

mode=$1

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

# Download data
if [ ! -f ./tmp/sample_video_id_path.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video_frame_extractor/sample_video_id_path.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video_frame_extractor/sample_videos.tgz
    
    mkdir tmp/

    tar -zxvf sample_videos.tgz  -C ./tmp/
    rm -rf sample_videos.tgz

    mv *.txt tmp/
fi

# 平均抽取frame_num帧

if [ "$mode" = "base64" ]; then
    # extract video frame, save images by image base64
    python examples/video_frame_extractor/main.py \
        --tables=./tmp/sample_video_id_path.txt \
        --video_root_dir=./tmp/sample_videos \
        --frame_num=3 \
        --outputs=./tmp/video_frames_output.txt

elif [ "$mode" = "path" ]; then
    # extract video frame, save images by image paths
    python examples/video_frame_extractor/main.py \
        --tables=./tmp/sample_video_id_path.txt \
        --video_root_dir=./tmp/sample_videos \
        --frame_num=3 \
        --outputs=./tmp/video_frames_output.txt \
        --frame_root_dir=./tmp/sample_frames \
        --enable_frame_path
fi
