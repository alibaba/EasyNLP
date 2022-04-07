export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
fi

if [ ! -f ./dev.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

mode=$2

if [ "$mode" = "train" ]; then
  easynlp \
    --mode $mode \
    --worker_gpu=1 \
    --tables=train.tsv,dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./classification_model \
    --learning_rate=3e-5  \
    --epoch_num=10  \
    --random_seed=42 \
    --save_checkpoint_steps=50 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --app_name=geep_classify \
    --user_defined_parameters='
        geep_exit_num=8
        pretrain_model_name_or_path=geep-base-uncased
    '
elif [ "$mode" = "evaluate" ]; then
  easynlp \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=dev.tsv \
  --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
  --first_sequence=sent1 \
  --second_sequence=sent2 \
  --label_name=label \
  --label_enumerate_values=0,1 \
  --checkpoint_dir=./classification_model \
  --sequence_length=128 \
  --micro_batch_size=32 \
  --app_name=geep_classify \
  --user_defined_parameters='
        geep_threshold=0.3
    '
elif [ "$mode" = "predict" ]; then
    easynlp \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=dev.tsv \
    --outputs=dev.pred.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --output_schema=predictions,probabilities,logits,output \
    --append_cols=label \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --checkpoint_path=./classification_model \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --app_name=geep_classify \
    --user_defined_parameters='
        geep_threshold=0.3 
    '
fi
