export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train_multilabel_zh.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/modelhub/nlu/general_news_classification/data/train.csv
fi

if [ ! -f ./dev_multilabel_zh.csv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/modelhub/nlu/general_news_classification/data/dev.csv:
fi

MASTER_ADDR=localhost
MASTER_PORT=6009
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

mode=$2

if [ "$mode" = "train" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
    --mode=$mode \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=content_seq:str:1,label:str:1 \
    --first_sequence=content_seq \
    --label_name=label \
    --label_enumerate_values=母婴,三农,科学,美文,科技,时尚,房产,美食,艺术,职场,健康,财经,国际,家居,娱乐,文化,教育,游戏,读书,动漫,体育,旅游,汽车,搞笑,健身,宠物,育儿 \
    --checkpoint_dir=./multi_label_classification_model \
    --learning_rate=3e-5  \
    --epoch_num=1  \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --train_batch_size=32 \
    --app_name=text_classify \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
    '

elif [ "$mode" = "evaluate" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=dev.csv \
  --input_schema=content_seq:str:1,label:str:1 \
  --first_sequence=content_seq \
  --label_name=label \
  --label_enumerate_values=母婴,三农,科学,美文,科技,时尚,房产,美食,艺术,职场,健康,财经,国际,家居,娱乐,文化,教育,游戏,读书,动漫,体育,旅游,汽车,搞笑,健身,宠物,育儿 \
  --checkpoint_dir=./multi_label_classification_model \
  --sequence_length=128 \
  --micro_batch_size=32 \
  --app_name=text_classify \
  --user_defined_parameters=''

elif [ "$mode" = "predict" ]; then

  python -m torch.distributed.launch $DISTRIBUTED_ARGS main.py \
  --mode=$mode \
  --worker_gpu=1 \
  --tables=dev.csv \
  --outputs=dev.pred.csv \
  --input_schema=content_seq:str:1,label:str:1 \
  --output_schema=predictions,probabilities,logits,output \
  --append_cols=label \
  --first_sequence=content_seq \
  --checkpoint_path=./multi_label_classification_model \
  --micro_batch_size=32 \
  --sequence_length=128 \
  --app_name=text_classify \
  --user_defined_parameters=''

fi
