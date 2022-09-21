export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./en_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/en_train.tsv
fi

if [ ! -f ./en_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/en_dev.tsv
fi

mode=$2

if [ "$mode" = "predict" ]; then
  
  easynlp \
    --app_name=sequence_generation \
    --mode $mode \
    --worker_gpu=1 \
    --tables=./en_dev.tsv  \
    --outputs=./en.preds.txt \
    --input_schema=title:str:1,content:str:1 \
    --output_schema=predictions,beams \
    --append_cols=title,content \
    --first_sequence=content \
    --checkpoint_dir=./finetuned_en_model \
    --micro_batch_size 32 \
    --sequence_length 512 \
    --user_defined_parameters 'language=en copy=false max_encoder_length=512 min_decoder_length=64 max_decoder_length=128 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "evaluate" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./en_dev.tsv  \
  --input_schema=title:str:1,content:str:1 \
  --output_schema=predictions,beams \
  --append_cols=title,content \
  --first_sequence=content \
  --second_sequence=title \
  --checkpoint_dir=./finetuned_en_model \
  --micro_batch_size 32 \
  --sequence_length 512 \
  --user_defined_parameters 'language=en copy=false max_encoder_length=512 min_decoder_length=64 max_decoder_length=128 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'

elif [ "$mode" = "train" ]; then

  easynlp \
  --app_name=sequence_generation \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=./en_train.tsv,./en_dev.tsv  \
  --input_schema=title:str:1,content:str:1 \
  --first_sequence=content \
  --second_sequence=title \
  --label_name=title \
  --checkpoint_dir=./finetuned_en_model \
  --micro_batch_size=1 \
  --sequence_length=512 \
  --save_checkpoint_steps=2000 \
  --export_tf_checkpoint_type none \
  --user_defined_parameters 'language=en pretrain_model_name_or_path=alibaba-pai/pegasus-summary-generation-en copy=false max_encoder_length=512 min_decoder_length=64 max_decoder_length=128 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'


fi
