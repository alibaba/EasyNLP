export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./cn_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_train.tsv
fi

if [ ! -f ./cn_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_dev.tsv
fi

if [ ! -f ./config_ds_glm_large_generation.json ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/public/mg/config_ds_glm_large_generation.json
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/public/mg/config_ds_glm_10B_generation.json
fi

mode=$2
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
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
MEGATRON_PARAMETERS="--deepspeed \
            --deepspeed_config ./config_ds_glm_large_generation.json \
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

# MEGATRON_PARAMETERS is only valid for megatron models such as mg/glm-large-zh

if [ "$mode" = "predict" ]; then
  
  easynlp \
    --app_name=sequence_generation \
    --mode $mode \
    --worker_gpu=1 \
    --tables=./cn_dev.tsv  \
    --outputs=./cn.preds.txt \
    --input_schema=target:str:1,source:str:1 \
    --output_schema=predictions,beams \
    --append_cols=target,source \
    --first_sequence=source \
    --checkpoint_dir=./finetuned_zh_model \
    --micro_batch_size=32 \
    --sequence_length 512 \
    $MEGATRON_PARAMETERS \
    --user_defined_parameters 'copy=false language=zh max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./cn_dev.tsv  \
  --input_schema=target:str:1,source:str:1 \
  --output_schema=predictions,beams \
  --append_cols=target,source \
  --first_sequence=source \
  --second_sequence=target \
  --checkpoint_dir=./finetuned_zh_model \
  --micro_batch_size=32 \
  --sequence_length 512 \
  $MEGATRON_PARAMETERS \
  --user_defined_parameters 'copy=false language=zh max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=1'
  
elif [ "$mode" = "train" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./cn_train.tsv,./cn_dev.tsv  \
  --input_schema=target:str:1,source:str:1 \
  --first_sequence=source \
  --second_sequence=target \
  --label_name=target \
  --checkpoint_dir=./finetuned_zh_model \
  --micro_batch_size=16 \
  --sequence_length=512 \
  --epoch_num 3 \
  --save_checkpoint_steps=150 \
  --export_tf_checkpoint_type none \
  $MEGATRON_PARAMETERS \
  --user_defined_parameters 'pretrain_model_name_or_path=mg/glm-large-chinese language=zh copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

fi

# alibaba-pai/mt5-title-generation-zh
# hfl/randeng-summary-generation-base-zh
# hfl/randeng-summary-generation-large-zh
# alibaba-pai/randeng-title-generation-base-zh
# alibaba-pai/randeng-title-generation-large-zh
# mg/glm-large-zh
