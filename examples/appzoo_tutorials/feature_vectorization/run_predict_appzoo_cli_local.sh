export CUDA_VISIBLE_DEVICES=$1


if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

mode=$2

if [ "$mode" = "predict" ]; then

    easynlp \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=dev.tsv \
    --outputs=dev.pred.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --output_schema=pooler_output,first_token_output,all_hidden_outputs \
    --first_sequence=sent1 \
    --append_cols=label \
    --checkpoint_dir=./classification_model/ \
    --micro_batch_size=32 \
    --sequence_length=128 \
    --app_name=vectorization

fi
