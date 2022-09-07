export CUDA_VISIBLE_DEVICES=$1

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

MASTER_ADDR=localhost
MASTER_PORT=6007
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

# Download data
if [ ! -f ./tmp/VC_train_base64.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_train_base64.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_val_base64.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/video2text_generation/VC_test_base64.txt
    
    mkdir tmp/
    mv *.txt tmp/
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
mode=$2

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


  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=train \
    --tables=./tmp/VC_train_path.txt,./tmp/VC_val_path.txt \
    --input_schema=id:str:1,imgpath:str:1,text:str:1 \
    --first_sequence=imgpath \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_gen_model_pretrain \
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
        n_layer=12
        n_head=12
        n_embd=768
      ' 
      
# format of training file: id \t [image_base64_1, image_base64_2, ...., image_base64_n] \t text 
elif [ "$mode" = "pretrain" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=train \
    --tables=./tmp/VC_train_base64.txt,./tmp/VC_val_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_gen_model_pretrain \
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

elif [ "$mode" = "finetune" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=train \
    --tables=./tmp/VC_train_base64.txt,./tmp/VC_val_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_gen_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/v2t_gen_model_pretrain
        frame_num=4
        img_size=224
        img_len=256
        text_len=32
      ' 
      
elif [ "$mode" = "predict" ]; then
  rm -rf ./tmp/VC_outputs.txt
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=predict \
    --tables=./tmp/VC_test_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1 \
    --first_sequence=imgbase64 \
    --outputs=./tmp/VC_outputs.txt \
    --output_schema=idx,gen_text \
    --checkpoint_dir=./tmp/v2t_gen_model_finetune \
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
