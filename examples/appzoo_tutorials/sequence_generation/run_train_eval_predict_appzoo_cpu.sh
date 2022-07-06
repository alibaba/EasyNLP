
if [ ! -f ./cn_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_train.tsv
fi

if [ ! -f ./cn_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/generation/cn_dev.tsv
fi


cd ../../../

MODE=train
DIR=/apsarapangu/disk3/minghui.qmh/EasyNLP
TRAIN=$DIR/examples/application_tutorials/sequence_generation/cn_train.tsv
DEV=$DIR/examples/application_tutorials/sequence_generation/cn_dev.tsv

python $DIR/easynlp/appzoo/api.py \
  --distributed_backend=gloo \
  --mode=$MODE \
  --app_name=sequence_generation \
  --worker_gpu=0 \
  --worker_cpu=1 \
  --tables=$TRAIN,$DEV  \
  --input_schema=title_tokens:str:1,content_tokens:str:1 \
  --first_sequence=content_tokens \
  --second_sequence=title_tokens \
  --label_name=title_tokens \
  --checkpoint_dir=./finetuned_zh_model \
  --micro_batch_size=1 \
  --sequence_length=512 \
  --save_checkpoint_steps=150 \
  --export_tf_checkpoint_type none \
  --user_defined_parameters 'pretrain_model_name_or_path=alibaba-pai/mt5-title-generation-zh copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5'
