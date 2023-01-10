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

if [ "$mode" = "finetune" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/MUGE_train_text_imgbase64.tsv,./tmp/MUGE_val_text_imgbase64.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
    --learning_rate=4e-5 \
    --epoch_num=20 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=1000 \
    --sequence_length=288 \
    --micro_batch_size=32 \
    --app_name=latent_diffusion \
    --user_defined_parameters='
        pretrain_model_name_or_path=alibaba-pai/pai-diffusion-general-slim-zh
        size=256
        text_len=32
        img_len=256
      ' 


elif [ "$mode" = "predict" ]; then
  easynlp \
      --mode predict \
      --worker_gpu=1 \
      --tables=./tmp/MUGE_test.text.tsv \
      --input_schema=idx:str:1,text:str:1 \
      --output_schema=text \
      --outputs ./output_placeholder.tsv \
      --first_sequence=text \
      --checkpoint_dir=./tmp/finetune_model \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=latent_diffusion \
      --user_defined_parameters="n_samples=2 write_image=True image_prefix=./output/" 
fi