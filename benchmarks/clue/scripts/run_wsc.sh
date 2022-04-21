export CUDA_VISIBLE_DEVICES=$1

# Local training example
# cur_path=/tmp/EasyNLP
# cur_path=/apsarapangu/disk3/minghui.qmh/EasyNLP/
cur_path=/wjn/EasyNLP
mode=$2

cd ${cur_path}

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6010"

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/benchmarks/clue/main.py \
    --mode=$mode \
    --app_name=text_classify \
    --tables=tmp/train.tsv,tmp/dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./tmp/benchmarks/clue/ \
    --learning_rate=5e-5  \
    --epoch_num=50  \
    --random_seed=42 \
    --logging_steps=10 \
    --save_checkpoint_steps=34 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --user_defined_parameters='clue_name=clue task_name=wsc pretrain_model_name_or_path=bert-base-chinese'

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/benchmarks/clue/main_evaluate.py \
    --mode=$mode \
    --app_name=text_classify \
    --tables=tmp/train.tsv,tmp/dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./tmp/benchmarks/clue/ \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --user_defined_parameters='clue_name=clue task_name=wsc pretrain_model_name_or_path=bert-base-chinese'


elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/benchmarks/clue/main_predict.py \
      --mode=$mode \
      --app_name=text_classify \
      --tables=tmp/train.tsv,tmp/dev.tsv \
      --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
      --first_sequence=sent1 \
      --second_sequence=sent2 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --checkpoint_dir=./tmp/benchmarks/clue/ \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --user_defined_parameters='clue_name=clue task_name=wsc pretrain_model_name_or_path=bert-base-chinese'

fi