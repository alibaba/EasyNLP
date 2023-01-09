export CUDA_VISIBLE_DEVICES=$1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MASTER_ADDR=localhost
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
    --outputs=./chat_dev_gpt2.tsv \
    --input_schema=source:str:1,target:str:1 \
    --output_schema=predictions,beams \
    --append_cols=target,source \
    --first_sequence=source \
    --checkpoint_dir=./finetuned_zh_model-chat  \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --user_defined_parameters 'copy=false language=zh max_encoder_length=512 min_decoder_length=8 max_decoder_length=128 no_repeat_ngram_size=2 num_beams=15 num_return_sequences=5 num_beam_groups=5 diversity_penalty=1.0'

elif [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./chat_train.tsv,./chat_dev.tsv  \
    --input_schema=source:str:1,target:str:1 \
    --first_sequence=source \
    --second_sequence=target \
    --label_name=target \
    --checkpoint_dir=./finetuned_zh_model-chat \
    --learning_rate 1e-4  \
    --micro_batch_size 64 \
    --sequence_length 128 \
    --epoch_num 3   \
    --save_checkpoint_steps 500 \
    --export_tf_checkpoint_type none \
    --user_defined_parameters 'pretrain_model_name_or_path=alibaba-pai/gpt2-chitchat-zh language=zh copy=false max_encoder_length=128 min_decoder_length=4 max_decoder_length=128 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=./question_gen_dev_small.tsv \
    --input_schema=target:str:1,source:str:1 \
    --first_sequence=source \
    --second_sequence=target \
    --label_name=target \
    --checkpoint_dir=./finetuned_zh_model-bartbase-question \
    --micro_batch_size=32 \
    --sequence_length=512 \
    --export_tf_checkpoint_type none \
    --user_defined_parameters 'copy=false language=zh max_encoder_length=512 min_decoder_length=8 max_decoder_length=64 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

fi

