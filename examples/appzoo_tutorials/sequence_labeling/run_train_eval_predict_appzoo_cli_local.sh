export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/dev.csv
fi

if [ ! -f ./test.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/test.csv
fi

mode=$2

if [ "$mode" = "train" ]; then

  easynlp  \
    --mode=train \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=content:str:1,label:str:1 \
    --first_sequence=content \
    --label_name=label \
    --label_enumerate_values=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
    --checkpoint_dir=./labeling_model \
    --learning_rate=1e-4  \
    --epoch_num=1  \
    --logging_steps=100 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=64 \
    --app_name=sequence_labeling \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
    '

elif [ "$mode" = "evaluate" ]; then

  easynlp  \
    --mode=evaluate \
    --worker_gpu=1 \
    --tables=dev.csv \
    --input_schema=content:str:1,label:str:1 \
    --first_sequence=content \
    --label_name=label \
    --label_enumerate_values=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
    --checkpoint_path=./labeling_model \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --app_name=sequence_labeling

elif [ "$mode" = "predict" ]; then

  easynlp \
    --mode=predict \
    --worker_gpu=1 \
    --tables=test.csv \
    --outputs=test.pred.csv \
    --input_schema=content:str:1,label:str:1 \
    --first_sequence=content \
    --sequence_length=128 \
    --output_schema=output \
    --append_cols=label \
    --checkpoint_path=./labeling_model \
    --micro_batch_size=32 \
    --app_name=sequence_labeling

fi
