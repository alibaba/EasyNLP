export CUDA_VISIBLE_DEVICES=$1
mode=$2

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

# Download whl
if [ ! -f ./tmp/easynlp-0.0.5-py3-none-any.whl ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/geely_app/easynlp-0.0.5-py3-none-any.whl
fi

# Download data
if [ ! -f ./tmp/T2I_train.tsv ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_train.tsv
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_val.tsv
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_test.tsv
fi

# Download artist-large ckpt
if [ ! -f ./tmp/artist-large-zh.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/geely_app/artist-large-zh.tgz
    tar zxvf ./tmp/artist-large-zh.tgz -C ./tmp
fi

pip install ./tmp/easynlp-0.0.5-py3-none-any.whl 

if [ "$mode" = "pretrain" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/T2I_train.tsv,./tmp/T2I_val.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/artist_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/artist-large-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
    '


elif [ "$mode" = "finetune" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/T2I_train.tsv,./tmp/T2I_val.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/artist_model_pretrain
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 


elif [ "$mode" = "predict" ]; then
  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/T2I_test.tsv \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs.tsv \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune\
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
        max_generated_num=4
      '
fi