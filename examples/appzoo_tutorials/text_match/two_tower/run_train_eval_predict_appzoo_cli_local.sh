export CUDA_VISIBLE_DEVICES=$1

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/tutorials/ez_text_match/afqmc_public/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/tutorials/ez_text_match/afqmc_public/dev.csv
fi

mode=$2

if [ "$mode" = "train" ]; then

  easynlp \
    --mode $mode \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=example_id:str:1,sent1:str:1,sent2:str:1,label:str:1,cate:str:1,score:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./text_match_two_tower_model_dir \
    --learning_rate=3e-5  \
    --epoch_num=1  \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --train_batch_size=32 \
    --app_name=text_match \
    --user_defined_parameters='
        pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext
        two_tower=True
        loss_type=hinge_loss
        margin=0.45
        gamma=32
    '

fi
