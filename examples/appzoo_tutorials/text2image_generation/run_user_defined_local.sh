export CUDA_VISIBLE_DEVICES=$1

# Local training example
cur_path=$PWD/../../../
cd ${cur_path}

MASTER_ADDR=localhost
MASTER_PORT=6007
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0


if [ ! -f ./tmp/T2I_train.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_train.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_val.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_test.txt
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

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=train \
    --tables=./tmp/T2I_train.txt,./tmp/T2I_val.txt \
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
        tokenizer_name_or_path=hfl/chinese-roberta-wwm-ext
        vqgan_ckpt_path=./tmp/vqgan_f16_16384.bin
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 


elif [ "$mode" = "finetune" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=train \
    --tables=./tmp/T2I_train.txt,./tmp/T2I_val.txt \
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
        tokenizer_name_or_path=hfl/chinese-roberta-wwm-ext
        pretrain_model_name_or_path=artist-base-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 


elif [ "$mode" = "predict" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=predict \
    --tables=./tmp/T2I_test.txt \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs.txt \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        tokenizer_name_or_path=hfl/chinese-roberta-wwm-ext
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      '
fi
