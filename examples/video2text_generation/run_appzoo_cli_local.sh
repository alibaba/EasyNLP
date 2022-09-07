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
if [ ! -f ./tmp/IC_train_base64.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_train_base64.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_val_base64.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_test_base64.txt
    mv *.txt tmp/
fi

# run script
# pretrain from scratch
# format of training file: id \t [image_path_1, image_path_2, ......, image_path_n] \t text
if [ "$mode" = "pretrain_local_path" ]; then
  # Download data
  if [ ! -f ./tmp/VC_train_path.txt ]; then
      wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_train_path.txt
      wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_val_path.txt
      wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/frames.tgz
    
      mkdir tmp/
      mv *.txt tmp/

      mkdir tmp/sample_frame_images/
      tar -zxvf frames.tgz  -C ./tmp/sample_frame_images/
      rm -rf frames.tgz
  fi

  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/VC_train_path.txt,./tmp/VC_val_path.txt \
    --input_schema=id:str:1,imgpath:str:1,text:str:1 \
    --first_sequence=imgpath \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        enable_img_path=True 
        img_root_dir=./tmp/sample_frame_images/
        frame_num=4
        vit_ckpt_path=ViT-L/14
        img_size=224
        img_len=256
        text_len=32
        pretrain_model_name_or_path=bert-base-chinese
        block_size=288
        n_layer=24
        n_head=16
        n_embd=1024
      '

# pretrain from scratch
# format of training file: id \t [image_base64_1, image_base64_2, ...., image_base64_n] \t text 
elif [ "$mode" = "pretrain" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/VC_train_base64.txt,./tmp/VC_val_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        frame_num=4
        vit_ckpt_path=ViT-L/14
        img_size=224
        img_len=256
        text_len=32
        pretrain_model_name_or_path=bert-base-chinese
        block_size=288
        n_layer=12
        n_head=12
        n_embd=768
      '

# finetune
elif [ "$mode" = "finetune" ]; then
  easynlp \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/VC_train_base64.txt,./tmp/VC_val_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        frame_num=4
        pretrain_model_name_or_path=./tmp/v2t_model_pretrain
        img_size=224
        img_len=256
        text_len=32
      ' 

# predict
elif [ "$mode" = "predict" ]; then
  rm -rf ./tmp/VC_outputs.txt
  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/VC_test_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --outputs=./tmp/VC_outputs.txt \
    --output_schema=idx,gen_text \
    --checkpoint_dir=./tmp/v2t_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        frame_num=4
        img_size=224
        text_len=32
        img_len=256
        max_generated_num=4
      '
fi
