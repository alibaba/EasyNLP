export CUDA_VISIBLE_DEVICES=$1

# Local training example
cur_path=$PWD/../../../
cd ${cur_path}

MASTER_ADDR=localhost
MASTER_PORT=6007
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=train \
    --tables=/apsarapangu/disk2/hexi.ltt/data/MUGE/ECommerce-T2I/T2I_val_text_imgbase64.txt,/apsarapangu/disk2/hexi.ltt/data/MUGE/ECommerce-T2I/T2I_ceshi_text_imgbase64_100.txt \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=/apsarapangu/disk1/hexi.ltt/ckpt/easynlp-taming-ceshi \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        tokenizer_name_or_path=/apsarapangu/disk2/hexi.ltt/ckpt/chinese-roberta-wwm-ext
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
        vqgan_ckpt_path=/apsarapangu/disk2/hexi.ltt/ckpt/vqgan/imagenet_f16_16384/checkpoints/last.ckpt
      ' 

elif [ "$mode" = "predict" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=predict \
    --tables=/apsarapangu/disk2/hexi.ltt/data/MUGE/ECommerce-T2I/T2I_test.text.tsv \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/outputs.txt \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=/apsarapangu/disk1/hexi.ltt/ckpt/easynlp-taming-ceshi/ \
    --sequence_length=288 \
    --micro_batch_size=32 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        tokenizer_name_or_path=/apsarapangu/disk2/hexi.ltt/ckpt/chinese-roberta-wwm-ext/
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
        vqgan_ckpt_path=/apsarapangu/disk2/hexi.ltt/ckpt/vqgan/imagenet_f16_16384/checkpoints/last.ckpt
      '
fi
