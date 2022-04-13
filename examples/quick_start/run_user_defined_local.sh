export CUDA_VISIBLE_DEVICES=0

if [ ! -f ./tmp/train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train_toy.tsv
fi

cd ../../
python setup.py develop

cd examples/quick_start/
python main.py \
  --mode train \
  --tables=train_toy.tsv \
  --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
  --first_sequence=sent1 \
  --label_name=label \
  --label_enumerate_values=0,1 \
  --checkpoint_dir=./tmp/ \
  --epoch_num=1  \
  --app_name=text_classify \
  --user_defined_parameters='pretrain_model_name_or_path=bert-tiny-uncased'
