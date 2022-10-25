export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train_multilabel_zh.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train_multilabel_zh.csv
fi

if [ ! -f ./dev_multilabel_zh.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev_multilabel_zh.csv
fi

mode=$2

if [ "$mode" = "train" ]; then

  easynlp \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=train_multilabel_zh.csv,dev_multilabel_zh.csv \
    --input_schema=content_seq:str:1,label:str:1 \
    --first_sequence=content_seq \
    --label_name=label \
    --label_enumerate_values=体积大小,外观,制热范围,制热效果,衣物烘干,味道,产品功耗,滑轮提手,声音 \
    --checkpoint_dir=./multi_label_classification_model \
    --learning_rate=3e-5  \
    --epoch_num=20  \
    --random_seed=42 \
    --save_checkpoint_steps=50 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --app_name=text_classify \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
        multi_label=True
    '

elif [ "$mode" = "evaluate" ]; then

  easynlp \
      --mode=$mode \
      --worker_gpu=1 \
      --tables=dev_multilabel_zh.csv \
      --input_schema=content_seq:str:1,label:str:1 \
      --first_sequence=content_seq \
      --label_name=label \
      --label_enumerate_values=体积大小,外观,制热范围,制热效果,衣物烘干,味道,产品功耗,滑轮提手,声音 \
      --checkpoint_dir=./multi_label_classification_model \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=text_classify \
      --user_defined_parameters='
        multi_label=True
      '


elif [ "$mode" = "predict" ]; then

 easynlp \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=dev_multilabel_zh.csv \
    --outputs=dev_multilabel_zh.pred.csv \
    --input_schema=content_seq:str:1,label:str:1 \
    --output_schema=predictions,probabilities,logits,output \
    --append_cols=label \
    --first_sequence=content_seq \
    --checkpoint_path=./multi_label_classification_model \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --app_name=text_classify \
    --user_defined_parameters='
        multi_label=True
      '

fi