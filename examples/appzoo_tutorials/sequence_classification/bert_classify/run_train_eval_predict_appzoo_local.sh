
if [ ! -f ./train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
fi

if [ ! -f ./dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

MODE=train
DIR=/apsarapangu/disk3/minghui.qmh/EasyNLP
TRAIN=$DIR/examples/application_tutorials/sequence_classification/bert_classify/train.tsv
DEV=$DIR/examples/application_tutorials/sequence_classification/bert_classify/dev.tsv

DEVICE=$1
cd $DIR

if [ $DEVICE == 'cpu' ]; then
  export CUDA_VISIBLE_DEVICES=-1
  SCRIPT=examples/application_tutorials/sequence_classification/bert_classify/main.py
  python ./$SCRIPT \
      --mode $MODE \
      --worker_gpu=0 \
      --worker_cpu=1 \
      --tables=$TRAIN,$DEV \
      --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
      --first_sequence=sent1 \
      --second_sequence=sent2 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --checkpoint_dir=./tmp/classification_model \
      --learning_rate=3e-5  \
      --epoch_num=3  \
      --random_seed=42 \
      --save_checkpoint_steps=50 \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=text_classify \
      --user_defined_parameters='  pretrain_model_name_or_path=bert-base-uncased'
else 
  export CUDA_VISIBLE_DEVICES=5
  MASTER_ADDR=localhost
  MASTER_PORT=6010
  GPUS_PER_NODE=1
  NNODES=1
  NODE_RANK=0

  cd $DIR
  DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
  SCRIPT=examples/application_tutorials/sequence_classification/bert_classify/main.py
  python -m torch.distributed.launch ./$SCRIPT \
      --mode $MODE \
      --worker_gpu=1 \
      --tables=$TRAIN,$DEV \
      --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
      --first_sequence=sent1 \
      --second_sequence=sent2 \
      --label_name=label \
      --label_enumerate_values=0,1 \
      --checkpoint_dir=./tmp/classification_model \
      --learning_rate=3e-5  \
      --epoch_num=3  \
      --random_seed=42 \
      --save_checkpoint_steps=50 \
      --sequence_length=128 \
      --micro_batch_size=32 \
      --app_name=text_classify \
      --user_defined_parameters='  pretrain_model_name_or_path=bert-base-uncased'
fi


  