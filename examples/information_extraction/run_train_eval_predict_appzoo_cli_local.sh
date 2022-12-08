export CUDA_VISIBLE_DEVICES=$1

mode=$2

if [ "$mode" = "train" ]; then

  easynlp
    --mode $mode \
    --tables=train.tsv,dev.tsv \
    --input_schema=id:str:1,instruction:str:1,start:str:1,end:str:1,target:str:1 \
    --worker_gpu=4 \
    --app_name=information_extraction \
    --sequence_length=512 \
    --weight_decay=0.0 \
    --micro_batch_size=4 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5 \
    --user_defined_parameters='pretrain_model_name_or_path=hfl/macbert-large-zh' \
    --save_checkpoint_steps=500 \
    --gradient_accumulation_steps=8 \
    --epoch_num=3  \
    --learning_rate=2e-05  \
    --random_seed=42

elif [ "$mode" = "evaluate" ]; then

  easynlp
    --mode $mode \
    --tables=dev.tsv \
    --input_schema=id:str:1,instruction:str:1,start:str:1,end:str:1,target:str:1 \
    --worker_gpu=4 \
    --app_name=information_extraction \
    --sequence_length=512 \
    --weight_decay=0.0 \
    --micro_batch_size=4 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5

elif [ "$mode" = "predict" ]; then

  easynlp
    --mode=$mode \
    --tables=predict_input_EE.tsv,predict_output_EE.tsv \
    --input_schema=id:str:1,scheme:str:1,content:str:1 \
    --output_schema=id,content,q_and_a \
    --worker_gpu=4 \
    --app_name=information_extraction \
    --sequence_length=512 \
    --weight_decay=0.0 \
    --micro_batch_size=4 \
    --checkpoint_dir=./information_extraction_model/ \
    --data_threads=5 \
    --user_defined_parameters='task=EE'
fi