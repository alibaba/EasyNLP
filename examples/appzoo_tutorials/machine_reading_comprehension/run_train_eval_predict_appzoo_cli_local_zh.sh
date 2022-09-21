export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train_cmrc2018.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/train_cmrc2018.tsv
fi

if [ ! -f ./dev_cmrc2018.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/dev_cmrc2018.tsv
fi

if [ ! -f ./trial_cmrc2018.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/trial_cmrc2018.tsv
fi

mode=$2

if [ "$mode" = "train" ]; then

  easynlp \
    --mode=$mode \
    --app_name=machine_reading_comprehension \
    --worker_gpu=1 \
    --tables=train_cmrc2018.tsv,dev_cmrc2018.tsv \
    --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
    --first_sequence=question_text \
    --second_sequence=context_text \
    --sequence_length=384 \
    --checkpoint_dir=./cmrc_model_dir \
    --learning_rate=3.5e-5 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=600 \
    --train_batch_size=16 \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/macbert-base-zh 
        language=zh
        answer_name=answer_text
        qas_id=qas_id
        start_position_name=start_position_character
        doc_stride=128
        max_query_length=64
    '

elif [ "$mode" = "evaluate" ]; then

  easynlp \
    --mode=$mode \
    --app_name=machine_reading_comprehension \
    --worker_gpu=1 \
    --tables=dev_cmrc2018.tsv \
    --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
    --first_sequence=question_text \
    --second_sequence=context_text \
    --sequence_length=384 \
    --checkpoint_dir=./cmrc_model_dir \
    --micro_batch_size=16 \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/macbert-base-zh 
        language=zh
        qas_id=qas_id
        answer_name=answer_text
        start_position_name=start_position_character
        doc_stride=128
        max_query_length=64
    '

elif [ "$mode" = "predict" ]; then

    easynlp \
    --mode=$mode \
    --app_name=machine_reading_comprehension \
    --worker_gpu=1 \
    --tables=dev_cmrc2018.tsv \
    --outputs=dev_cmrc.pred.csv \
    --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
    --output_schema=unique_id,best_answer,query,context \
    --first_sequence=question_text \
    --second_sequence=context_text \
    --sequence_length=384 \
    --checkpoint_dir=./cmrc_model_dir \
    --micro_batch_size=256 \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/macbert-base-zh 
        language=zh
        qas_id=qas_id
        answer_name=answer_text
        start_position_name=start_position_character
        max_query_length=64
        max_answer_length=30
        doc_stride=128
        n_best_size=10
        output_answer_file=dev_cmrc.ans.csv
    '

fi
