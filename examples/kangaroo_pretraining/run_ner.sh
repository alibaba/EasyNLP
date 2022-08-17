export CUDA_VISIBLE_DEVICES=0

# Local training example
basepath=$PWD

cur_path=$basepath/../../
cd ${cur_path}

if [ ! -f tmp/finData ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/finData.tar.gz
  tar -zxvf finData.tar.gz
  rm -rf finData.tar.gz
  mkdir tmp/
  mv finData tmp/
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
python -m torch.distributed.launch $DISTRIBUTED_ARGS $basepath/kangaroo_ner.py \
  --mode train \
  --tables tmp/financial_ner/train.tsv,tmp/financial_ner/dev.tsv \
  --input_schema content:str:1,label:str:1 \
  --first_sequence content \
  --label_name label \
  --label_enumerate_values B-ORG,B-PER,B-POS,I-ORG,I-PER,I-POS,O \
  --checkpoint_dir ./tmp/kangaroo_fin_ner_model \
  --learning_rate 2e-5 \
  --epoch_num 2 \
  --random_seed 42 \
  --logging_steps 1 \
  --save_checkpoint_steps 50 \
  --sequence_length 128 \
  --micro_batch_size 16 \
  --app_name sequence_labeling \
  --use_amp \
  --user_defined_parameters "pretrain_model_name_or_path=alibaba-pai/pai-kangaroo-fin-base-chinese entity_file=tmp/finData/finEntity.csv rel_file=tmp/finData/finRelation.csv concept_emb_file=tmp/finData/finConceptEmbedding.npy samples_file=tmp/finData/finSamples4Level.npy"
