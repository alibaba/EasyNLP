export CUDA_VISIBLE_DEVICES=$1


if [ ! -f ./train_part.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ie/train_part.tsv
fi

if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ie/dev.tsv
fi

if [ ! -f ./predict_input_EE.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ie/predict_input_EE.tsv
fi

if [ ! -f ./predict_input_NER.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ie/predict_input_NER.tsv
fi

mode=$2

if [ "$mode" = "train" ]; then

  easynlp \
    --mode $mode \
    --tables=train_part.tsv,dev.tsv \
    --input_schema=id:str:1,instruction:str:1,start:str:1,end:str:1,target:str:1 \
    --worker_gpu=1 \
    --app_name=information_extraction \
    --sequence_length=128 \
    --weight_decay=0.0 \
    --micro_batch_size=8 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5 \
    --user_defined_parameters='pretrain_model_name_or_path=hfl/macbert-large-zh' \
    --save_checkpoint_steps=50 \
    --gradient_accumulation_steps=2 \
    --epoch_num=1  \
    --learning_rate=2e-05  \
    --random_seed=42

elif [ "$mode" = "evaluate" ]; then

  easynlp \
    --mode $mode \
    --tables=dev.tsv \
    --input_schema=id:str:1,instruction:str:1,start:str:1,end:str:1,target:str:1 \
    --worker_gpu=1 \
    --app_name=information_extraction \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5

elif [ "$mode" = "predict" ]; then

  easynlp \
    --mode=$mode \
    --tables=predict_input_EE.tsv \
    --outputs=predict_output_EE.tsv \
    --input_schema=id:str:1,scheme:str:1,content:str:1 \
    --outputs=predict_output_EE.tsv \
    --output_schema=id,content,q_and_a \
    --worker_gpu=1 \
    --app_name=information_extraction \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5 \
    --user_defined_parameters='task=EE'

  easynlp \
    --mode=$mode \
    --tables=predict_input_NER.tsv \
    --outputs=predict_output_NER.tsv \
    --input_schema=id:str:1,scheme:str:1,content:str:1 \
    --output_schema=id,content,q_and_a \
    --worker_gpu=1 \
    --app_name=information_extraction \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5 \
    --user_defined_parameters='task=NER'
fi

#mode=train处，目前使用的是部分训练数据，如果需要使用全部训练数据，请将train_part.tsv修改为train.tsv，save_checkpoint_steps=50修改为save_checkpoint_steps=500
