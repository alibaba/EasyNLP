export CUDA_VISIBLE_DEVICES=$1
mode=$2

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

# Download data
if [ ! -f ./tmp/MUGE_train_text_imgbase64.tsv ]; then
    wget  -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/MUGE_train_text_imgbase64.tsv
    wget  -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/MUGE_val_text_imgbase64.tsv
    wget  -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/MUGE_test.text.tsv
fi

if [ "$mode" = "pretrain" ]; then
easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/MUGE_train_text_imgbase64.tsv,./tmp/MUGE_val_text_imgbase64.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/continue_pretrain_model/ \
    --learning_rate=4e-5 \
    --epoch_num=40 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=1000 \
    --sequence_length=288 \
    --micro_batch_size=16 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=alibaba-pai/pai-painter-base-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
    ' 

elif [ "$mode" = "finetune" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/MUGE_train_text_imgbase64.tsv,./tmp/MUGE_val_text_imgbase64.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
    --learning_rate=4e-5 \
    --epoch_num=40 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=1000 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/continue_pretrain_model/
        size=256
        text_len=32
        img_len=256
    '


elif [ "$mode" = "predict" ]; then
  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/MUGE_test.text.tsv \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs.tsv \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
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