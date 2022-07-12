export CUDA_VISIBLE_DEVICES=$1

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

MASTER_ADDR=localhost
MASTER_PORT=6007
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0


if [ ! -f ./tmp/IC_train.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_image2text/IC_train.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_image2text/IC_val.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_image2text/IC_test.txt
    mkdir tmp/
    mv *.txt tmp/
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
mode=$2


if [ "$mode" = "pretrain" ]; then
  if [ ! -f ./tmp/vqgan_f16_16384.bin ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/vqgan_f16_16384.bin
    mv vqgan_f16_16384.bin tmp/
  fi

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/image2text_generation/main.py \
    --mode=train \
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


elif [ "$mode" = "finetune" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/image2text_generation/main.py \
    --mode=train \
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
        pretrain_model_name_or_path=artist-i2t-large-zh
        img_size=256
        img_len=256
        text_len=32
      ' 


elif [ "$mode" = "predict" ]; then
  rm -rf ./tmp/IC_outputs.txt
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/image2text_generation/main.py \
    --mode=predict \
    --tables=./tmp/IC_test.txt \
    --input_schema=idx:str:1,imgbase64:str:1 \
    --first_sequence=imgbase64 \
    --outputs=./tmp/IC_outputs.txt \
    --output_schema=idx,gen_text \
    --checkpoint_dir=./tmp/artist_i2t_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=image2text_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./tmp/artist_i2t_model_finetune
        img_size=256
        text_len=32
        img_len=256
      '
fi