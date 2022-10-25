export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train_squad.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/train_squad.tsv
fi

if [ ! -f ./dev_squad.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/dev_squad.tsv
fi

MASTER_ADDR=localhost
MASTER_PORT=6009
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then

  python main.py \
    --mode=$mode \
    --app_name=machine_reading_comprehension \
    --worker_gpu=1 \
    --tables=train_squad.tsv,dev_squad.tsv \
    --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
    --first_sequence=question_text \
    --second_sequence=context_text \
    --sequence_length=384 \
    --checkpoint_dir=./squad_model_dir \
    --learning_rate=3.5e-5 \
    --epoch_num=10 \
    --random_seed=42 \
    --save_checkpoint_steps=1000 \
    --train_batch_size=32 \
    --user_defined_parameters='
        pretrain_model_name_or_path=bert-base-uncased
        language=en
        answer_name=answer_text
        qas_id=qas_id
        start_position_name=start_position_character
        doc_stride=128
        max_query_length=64
    '

elif [ "$mode" = "evaluate" ]; then

  python main.py \
    --mode=$mode \
    --app_name=machine_reading_comprehension \
    --worker_gpu=1 \
    --tables=dev_squad.tsv \
    --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
    --first_sequence=question_text \
    --second_sequence=context_text \
    --sequence_length=384 \
    --checkpoint_dir=./squad_model_dir \
    --micro_batch_size=32 \
    --user_defined_parameters='
        pretrain_model_name_or_path=bert-base-uncased
        language=en
        qas_id=qas_id
        answer_name=answer_text
        start_position_name=start_position_character
        doc_stride=128
        max_query_length=64
    '

elif [ "$mode" = "predict" ]; then

  python main.py \
    --mode=$mode \
    --app_name=machine_reading_comprehension \
    --worker_gpu=1 \
    --tables=dev_squad.tsv \
    --outputs=dev.pred.csv \
    --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
    --output_schema=unique_id,best_answer,query,context \
    --first_sequence=question_text \
    --second_sequence=context_text \
    --sequence_length=384 \
    --checkpoint_dir=./squad_model_dir \
    --micro_batch_size=256 \
    --user_defined_parameters='
        pretrain_model_name_or_path=bert-base-uncased
        language=en
        qas_id=qas_id
        answer_name=answer_text
        start_position_name=start_position_character
        max_query_length=64
        max_answer_length=30
        doc_stride=128
        n_best_size=10
        output_answer_file=dev.ans.csv
    '

fi
