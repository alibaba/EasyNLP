export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./cn_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_train.tsv
fi

if [ ! -f ./cn_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_dev.tsv
fi

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="./hostfile"
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=2

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
DISTRIBUTED_ARGS="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH} --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}"

mode=$2

MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-position-embeddings 1024 \
            --tokenizer-type ChineseSPTokenizer \
            --fix-command-token"

TRAIN_ARGS="--lr-decay-style linear \
            --label-smoothing 0.1"

COMMON_ARGS="--save-interval 10000 \
             --log-interval 50 \
             --eval-interval 200 \
             --eval-iters 100"

TASK_ARGS="--length-penalty 0.7 \
           --select-topk \
           --eval-batch-size 1"

TASK_NAME=chinesegen
ALL_MG_PARA="--deepspeed \
            --deepspeed_config examples/appzoo_tutorials/sequence_generation/config_glm_large_generation.json \
            --finetune \
            --task ${TASK_NAME} \
            --data-dir ./ \
            --checkpoint-activations \
            --no-load-lr-scheduler \
            --num-workers 1 \
            --model-parallel-size 1 \
            $MODEL_ARGS \
            $TRAIN_ARGS \
            $COMMON_ARGS \
            $TASK_ARGS \
            --fp16 \
            --overwrite
            "

if [ "$mode" = "predict" ]; then
  
  run_cmd="$DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode $mode \
    --worker_gpu=1 \
    --tables=./cn_dev.tsv  \
    --outputs=./cn_dev.preds.txt \
    --input_schema=title:str:1,content:str:1 \
    --output_schema=predictions,beams \
    --append_cols=title,content \
    --first_sequence=content \
    --checkpoint_dir=./finetuned_zh_model \
    --micro_batch_size=32 \
    --sequence_length=512 \
    $ALL_MG_PARA \
    --user_defined_parameters 'copy=false language=zh max_encoder_length=512 min_decoder_length=12 max_decoder_length=40 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'
    "
  echo ${run_cmd}
  eval ${run_cmd}
elif [ "$mode" = "train" ]; then

  run_cmd="$DISTRIBUTED_ARGS  examples/appzoo_tutorials/sequence_generation/main.py \
    --app_name=sequence_generation \
    --mode=$mode \
    --mg_model \
    --worker_gpu 1 \
    --tables=./cn_train.tsv,./cn_dev.tsv  \
    --input_schema=title_tokens:str:1,content_tokens:str:1 \
    --first_sequence=content_tokens \
    --second_sequence=title_tokens \
    --label_name=title_tokens \
    --checkpoint_dir=./finetuned_zh_model \
    --learning_rate 5e-5  \
    --micro_batch_size 8 \
    --sequence_length 512 \
    --epoch_num 1   \
    --save_checkpoint_steps 150 \
    --export_tf_checkpoint_type none \
    $ALL_MG_PARA \
    --user_defined_parameters 'pretrain_model_name_or_path=mg/glm-large-chinese language=zh copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=40 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'
    "
  echo ${run_cmd}
  eval ${run_cmd}
# alibaba-pai/mt5-title-generation-zh
# hfl/randeng-523M-Summary-Chinese
# hfl/randeng-238M-Summary-Chinese
# alibaba-pai/randeng-523M-Summary-Chinese-tuned
# alibaba-pai/randeng-238M-Summary-Chinese-tuned
# mg/glm-large-chinese

elif [ "$mode" = "evaluate" ]; then

  run_cmd="$DISTRIBUTED_ARGS examples/appzoo_tutorials/sequence_generation/main.py \
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
    $ALL_MG_PARA \
    --user_defined_parameters 'copy=false language=zh max_encoder_length=512 min_decoder_length=12 max_decoder_length=50 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'
    "
  echo ${run_cmd}
  eval ${run_cmd}
fi
