export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./cn_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_train.tsv
fi

if [ ! -f ./cn_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_dev.tsv
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
    --tables=./cn_dev.tsv  \
    --outputs=./cn.preds.txt \
    --input_schema=title:str:1,content:str:1,title_tokens:str:1,content_tokens:str:1,tag:str:1 \
    --output_schema=predictions,beams \
    --append_cols=title,content,tag \
    --first_sequence=content \
    --checkpoint_dir=./finetuned_zh_model/ \
    --micro_batch_size=32 \
    --sequence_length=512 \
    --user_defined_parameters 'copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=40 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./cn_train.tsv,./cn_dev.tsv  \
    --input_schema=title_tokens:str:1,content_tokens:str:1 \
    --first_sequence=content_tokens \
    --second_sequence=title_tokens \
    --label_name=title_tokens \
    --checkpoint_dir=./finetuned_zh_model \
    --learning_rate=5e-5  \
    --micro_batch_size=8 \
    --sequence_length=512 \
    --epoch_num=1  \
    --save_checkpoint_steps=150 \
    --export_tf_checkpoint_type none \
    --user_defined_parameters 'pretrain_model_name_or_path=hfl/randeng-238M-Summary-Chinese copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=40 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

# alibaba-pai/mt5-title-generation-zh
# hfl/bloom-350m
# hfl/randeng-523M-Summary-Chinese
# hfl/randeng-238M-Summary-Chinese

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./cn_dev.tsv \
    --input_schema=title_tokens:str:1,content_tokens:str:1 \
    --first_sequence=content_tokens \
    --second_sequence=title_tokens \
    --label_name=title_tokens \
    --checkpoint_dir=./finetuned_zh_model \
    --micro_batch_size=16 \
    --sequence_length=512 \
    --export_tf_checkpoint_type none \
    --user_defined_parameters 'copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=50 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

fi
