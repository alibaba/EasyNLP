export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./chat_train.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/chat_train.tsv
fi

if [ ! -f ./chat_dev.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/chat_dev.tsv
fi

mode=$2

if [ "$mode" = "predict" ]; then
  
  easynlp \
    --app_name=sequence_generation \
    --mode $mode \
    --worker_gpu=1 \
    --tables=./chat_train.tsv,./chat_dev.tsv  \
    --outputs=./chat.preds.txt \
    --input_schema=first_sen:str:1,second_sen:str:1 \
    --output_schema=predictions,beams \
    --append_cols=second_sen \
    --first_sequence=first_sen \
    --checkpoint_dir=./finetuned_chat_model \
    --micro_batch_size=32 \
    --sequence_length 512 \
    --user_defined_parameters 'service=chat copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=256 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./chat_train.tsv,./chat_dev.tsv  \
  --input_schema=first_sen:str:1,second_sen:str:1 \
  --output_schema=predictions,beams \
  --append_cols=second_sen \
  --first_sequence=first_sen \
  --second_sequence=second_sen \
  --checkpoint_dir=./finetuned_chat_model \
  --micro_batch_size=32 \
  --sequence_length 512 \
  --user_defined_parameters 'service=chat copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=256 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'
  
elif [ "$mode" = "train" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./chat_train.tsv,./chat_dev.tsv  \
  --input_schema=first_sen:str:1,second_sen:str:1 \
  --first_sequence=first_sen \
  --second_sequence=second_sen \
  --label_name=second_sen \
  --checkpoint_dir=./finetuned_chat_model \
  --micro_batch_size=8 \
  --sequence_length=512 \
  --save_checkpoint_steps=7000 \
  --export_tf_checkpoint_type none \
  --user_defined_parameters 'service=chat pretrain_model_name_or_path=alibaba-pai/gpt2-chitchat-zh copy=false max_encoder_length=512 min_decoder_length=2 max_decoder_length=256 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

fi
