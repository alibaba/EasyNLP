export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./cn_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/generation/cn_train.tsv
fi

if [ ! -f ./cn_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easynlp/tutorials/generation/cn_dev.tsv
fi

mode=$2

if [ "$mode" = "predict" ]; then
  
  easynlp \
    --app_name=sequence_generation \
    --mode $mode \
    --worker_gpu=1 \
    --tables=./cn_dev.tsv  \
    --outputs=./cn.preds.txt \
    --input_schema=title:str:1,content:str:1,title_tokens:str:1,content_tokens:str:1,tag:str:1 \
    --output_schema=predictions,beams \
    --append_cols=title_tokens,content,tag \
    --first_sequence=content_tokens \
    --checkpoint_dir=./finetuned_zh_model \
    --micro_batch_size=32 \
    --sequence_length 512 \
    --user_defined_parameters 'copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./cn_dev.tsv  \
  --input_schema=title:str:1,content:str:1,title_tokens:str:1,content_tokens:str:1,tag:str:1 \
  --output_schema=predictions,beams \
  --append_cols=title_tokens,content,tag \
  --first_sequence=content_tokens \
  --second_sequence=title_tokens \
  --checkpoint_dir=./finetuned_zh_model \
  --micro_batch_size=32 \
  --sequence_length 512 \
  --user_defined_parameters 'copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'
  

elif [ "$mode" = "train" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./cn_train.tsv,./cn_dev.tsv  \
  --input_schema=title_tokens:str:1,content_tokens:str:1 \
  --first_sequence=content_tokens \
  --second_sequence=title_tokens \
  --label_name=title_tokens \
  --checkpoint_dir=./finetuned_zh_model \
  --micro_batch_size=8 \
  --sequence_length=512 \
  --save_checkpoint_steps=150 \
  --export_tf_checkpoint_type none \
  --user_defined_parameters 'pretrain_model_name_or_path=alibaba-pai/mt5-title-generation-zh copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

fi
