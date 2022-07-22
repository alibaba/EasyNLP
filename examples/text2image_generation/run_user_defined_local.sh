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
if [ ! -f ./tmp/T2I_train.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_train.tsv
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_val.tsv
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_test.tsv
    mkdir tmp/
    mv *.tsv tmp/
fi

# Download artist-large ckpt
if [ ! -f ./tmp/artist-large-zh.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/geely_app/artist-large-zh.tgz
    tar zxvf ./tmp/artist-large-zh.tgz -C ./tmp
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
mode=$2


if [ "$mode" = "pretrain" ]; then
  if [ ! -f ./tmp/vqgan_f16_16384.bin ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/vqgan_f16_16384.bin
    mv vqgan_f16_16384.bin tmp/
  fi

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
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
        vqgan_ckpt_path=./tmp/vqgan_f16_16384.bin
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
        text_vocab_size=21128
        n_layer=12
        n_head=12
        n_embd=768
    '


elif [ "$mode" = "finetune" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
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
        pretrain_model_name_or_path=artist-base-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 


elif [ "$mode" = "predict" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/T2I_test.tsv \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs.tsv \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
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