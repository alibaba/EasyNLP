export CUDA_VISIBLE_DEVICES=$1
mode=$2

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

mkdir tmp

# Download whl
if [ ! -f ./tmp/pai_easynlp-0.0.7-py3-none-any.whl ]; then
   wget -P ./tmp/ https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/geely_app/image2text/pai_easynlp-0.0.7-py3-none-any.whl
fi
pip install ./tmp/pai_easynlp-0.0.7-py3-none-any.whl --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple 

# Download data
if [ ! -f ./tmp/IC_train.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/image2text_generation/IC_train.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/image2text_generation/IC_val.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/image2text_generation/IC_test.txt
    mv *.txt tmp/
fi


# pretrain from scratch
if [ "$mode" = "pretrain" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/IC_train.txt,./tmp/IC_val.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/i2t_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        enable_vqgan=True
        vqgan_ckpt_path=./tmp/vqgan_f16_16384.bin
        img_size=256
        img_len=256
        text_len=32
        text_tokenizer=bert-base-chinese
        vocab_size=37513
        img_vocab_size=16384
        text_vocab_size=21128
        block_size=288
        n_layer=24
        n_head=16
        n_embd=1024
      '

# finetune
elif [ "$mode" = "finetune" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/IC_train.txt,./tmp/IC_val.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/i2t_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        enable_vqgan=True
        pretrain_model_name_or_path=./tmp/i2t_model_pretrain
        img_size=256
        img_len=256
        text_len=32
      ' 

# predict
elif [ "$mode" = "predict" ]; then
  rm -rf ./tmp/IC_outputs.txt
  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/IC_train.txt,./tmp/IC_val.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --outputs=./tmp/IC_outputs.txt \
    --output_schema=idx,gen_text \
    --checkpoint_dir=./tmp/i2t_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        enable_vqgan=True
        img_size=256
        text_len=32
        img_len=256
        max_generated_num=4
      '
fi
