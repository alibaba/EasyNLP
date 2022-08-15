export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./chat_train.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/chat_train.tsv
fi

if [ ! -f ./chat_dev.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/chat_dev.tsv
fi

function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000)) #增加一个10位的数再求余
    echo $(($num%$max+$min))
}
rnd=$(rand 5000 9000)
MASTER_PORT=$rnd
MASTER_ADDR=localhost
# MASTER_PORT=6008
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "predict" ]; then
  
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode $mode \
    --worker_gpu=1 \
    --tables=./chat_dev.tsv  \
    --outputs=./chat.preds.txt \
    --input_schema=first_sen:str:1,second_sen:str:1 \
    --output_schema=predictions,beams \
    --append_cols=second_sen \
    --first_sequence=first_sen \
    --checkpoint_dir=./finetuned_chat_model/ \
    --micro_batch_size=32 \
    --sequence_length 128 \
    --user_defined_parameters 'service=chat copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=20 no_repeat_ngram_size=1 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./chat_dev.tsv  \
    --input_schema=first_sen:str:1,second_sen:str:1 \
    --output_schema=predictions,beams \
    --append_cols=second_sen \
    --first_sequence=first_sen \
    --second_sequence=second_sen \
    --checkpoint_dir=./finetuned_chat_model/ \
    --micro_batch_size=32 \
    --sequence_length=512 \
    --user_defined_parameters 'service=chat copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=20 no_repeat_ngram_size=1 num_beams=5 num_return_sequences=5'
  
elif [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./chat_train.tsv,./chat_dev.tsv  \
    --input_schema=first_sen:str:1,second_sen:str:1 \
    --first_sequence=first_sen \
    --second_sequence=second_sen \
    --label_name=second_sen \
    --checkpoint_dir=./finetuned_chat_model/ \
    --micro_batch_size=8 \
    --learning_rate 5e-5 \
    --sequence_length=256 \
    --save_checkpoint_steps=1000 \
    --export_tf_checkpoint_type none \
    --epoch_num 3 \
    --user_defined_parameters 'service=chat pretrain_model_name_or_path=alibaba-pai/gpt2-chitchat-zh copy=false max_encoder_length=512 min_decoder_length=2 max_decoder_length=40 no_repeat_ngram_size=1 num_beams=5 num_return_sequences=5'
# alibaba-pai/mt5-title-generation-zh
# hfl/bloom-350m
# alibaba-pai/gpt2-chitchat-zh
fi
