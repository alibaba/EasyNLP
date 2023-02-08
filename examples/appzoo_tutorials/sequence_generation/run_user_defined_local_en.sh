export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./en_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/en_train.tsv
fi

if [ ! -f ./en_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/en_dev.tsv
fi

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
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
    --tables=./en_dev.tsv  \
    --outputs=./en_dev.preds.txt \
    --input_schema=title:str:1,content:str:1 \
    --output_schema=predictions,beams \
    --append_cols=title,content \
    --first_sequence=content \
    --checkpoint_dir=/root/.easynlp/modelzoo/alibaba-pai/pegasus-summary-generation-en \
    --micro_batch_size 32 \
    --sequence_length 512 \
    --user_defined_parameters 'language=en copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=64 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./en_dev.tsv  \
    --input_schema=title:str:1,content:str:1 \
    --output_schema=predictions,beams \
    --append_cols=title,content \
    --first_sequence=content \
    --second_sequence=title \
    --checkpoint_dir=./finetuned_en_model/ \
    --micro_batch_size 32 \
    --sequence_length 512 \
    --user_defined_parameters 'language=en copy=false max_encoder_length=512 min_decoder_length=64 max_decoder_length=128 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./en_train.tsv,./en_dev.tsv  \
    --input_schema=title:str:1,content:str:1 \
    --first_sequence=content \
    --second_sequence=title \
    --label_name=title \
    --checkpoint_dir=./finetuned_en_model/ \
    --micro_batch_size=8 \
    --learning_rate 5e-5 \
    --sequence_length=512 \
    --epoch_num 1 \
    --save_checkpoint_steps=200 \
    --export_tf_checkpoint_type none \
    --user_defined_parameters 'language=en pretrain_model_name_or_path=alibaba-pai/pegasus-summary-generation-en language=en copy=false max_encoder_length=512 min_decoder_length=32 max_decoder_length=64 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

# alibaba-pai/pegasus-summary-generation-en
# hfl/brio-large-en

fi
