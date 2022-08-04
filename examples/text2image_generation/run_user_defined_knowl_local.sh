export CUDA_VISIBLE_DEVICES=$1

# Local training example
cur_path=$PWD/../../
cd ${cur_path}

MASTER_ADDR=localhost
MASTER_PORT=6007
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
mode=$2

# Download data
if [ ! -f ./tmp/T2I_train.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_train.tsv
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_val.tsv
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_test.tsv
    mkdir tmp/
    mv *.tsv tmp/
fi


if [ ! -f ./tmp/entity2id.txt ]; then
  wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/entity2id.txt
  # wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/entity2vec.pt
fi


# preprocess data
if [ "$mode" = "preprocess" ]; then
  python examples/text2image_generation/preprocess_data_knowl.py \
    --input_file ./tmp/T2I_train.tsv \
    --entity_map_file ./tmp/entity2id.txt \
    --output_file ./tmp/T2I_knowl_train.tsv

  python examples/text2image_generation/preprocess_data_knowl.py \
    --input_file ./tmp/T2I_val.tsv \
    --entity_map_file ./tmp/entity2id.txt \
    --output_file ./tmp/T2I_knowl_val.tsv

  python examples/text2image_generation/preprocess_data_knowl.py \
    --input_file ./tmp/T2I_test.tsv \
    --entity_map_file ./tmp/entity2id.txt \
    --output_file ./tmp/T2I_knowl_test.tsv


elif [ "$mode" = "finetune" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main_knowl.py \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/T2I_knowl_train.tsv,./tmp/T2I_knowl_val.tsv \
    --input_schema=idx:str:1,text:str:1,lex_id:str:1,pos_s:str:1,pos_e:str:1,token_len:str:1,imgbase64:str:1,  \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=2 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=alibaba-pai/pai-artist-knowl-base-zh
        entity_emb_path=./tmp/entity2vec.pt
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 


elif [ "$mode" = "predict" ]; then
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main_knowl.py \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/T2I_knowl_test.tsv \
    --input_schema=idx:str:1,text:str:1,lex_id:str:1,pos_s:str:1,pos_e:str:1,token_len:str:1, \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs_knowl.tsv \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        entity_emb_path=./tmp/entity2vec.pt
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
        max_generated_num=4
      '
fi