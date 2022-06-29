export CUDA_VISIBLE_DEVICES=0

# Local training example
basepath=$PWD
cur_path=$basepath/../../

cd ${cur_path}

if [ ! -f ./tmp/kbert_data ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/K-BERT/kbert_data.zip
  unzip kbert_data.zip
  rm -rf kbert_data.zip
  mkdir tmp/
  mv kbert_data tmp/
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
python -m torch.distributed.launch $DISTRIBUTED_ARGS $basepath/kbert_ner.py \
  --mode train \
  --tables tmp/kbert_data/financial_ner/train.tsv,tmp/kbert_data/financial_ner/dev.tsv \
  --input_schema content:str:1,label:str:1 \
  --first_sequence content \
  --label_name label\
  --label_enumerate_values B-ORG,B-PER,B-POS,I-ORG,I-PER,I-POS,O \
  --checkpoint_dir ./tmp/kbert_ner_model/ \
  --learning_rate 2e-5 \
  --epoch_num 2 \
  --random_seed 42 \
  --logging_steps 1 \
  --save_checkpoint_steps 50 \
  --sequence_length 128 \
  --micro_batch_size 10 \
  --app_name sequence_labeling \
  --use_amp \
  --user_defined_parameters "pretrain_model_name_or_path=kbert-base-chinese kg_file=tmp/kbert_data/kbert_kgs/HowNet.spo"

