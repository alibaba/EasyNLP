export CUDA_VISIBLE_DEVICES=$1
mode=$2

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

mkdir tmp

# Download whl
if [ ! -f ./tmp/easynlp-0.0.7-py3-none-any.whl ]; then
    wget -P ./tmp/ https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/geely_app/image2text/easynlp-0.0.7-py3-none-any.whl
fi
pip install ./tmp/easynlp-0.0.7-py3-none-any.whl 

# Download data
if [ ! -f ./tmp/IC_train.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_image2text/IC_train.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_image2text/IC_val.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_image2text/IC_test.txt
    mv *.txt tmp/
fi

# Download artist-large ckpt
if [ ! -f ./tmp/artist-i2t-large-zh.tgz ]; then
    wget -P ./tmp/ https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/geely_app/artist-i2t-large-zh.tgz
fi
tar zxvf ./tmp/artist-i2t-large-zh.tgz -C ./tmp/

if [ "$mode" = "pretrain" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/IC_train.txt,./tmp/IC_val.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/artist_i2t_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/artist-i2t-large-zh
        img_size=256
        text_len=32
        img_len=256
    '


elif [ "$mode" = "finetune" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/IC_train.txt,./tmp/IC_val.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/artist_i2t_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/artist_i2t_model_pretrain
        img_size=256
        text_len=32
        img_len=256
      ' 


elif [ "$mode" = "predict" ]; then
  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/IC_train.txt,./tmp/IC_val.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --outputs=./tmp/IC_outputs.txt \
    --output_schema=idx,imgbase64,gen_text \
    --checkpoint_dir=./tmp/artist_i2t_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        img_size=256
        text_len=32
        img_len=256
      '
fi