#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

# Cleanup environment
rm -rf $HOME/.easynlp_modelzoo
rm -rf $HOME/.cache/huggingface

export MKL_THREADING_LAYER=GNU

cd ../

pip uninstall easynlp -y

if [ ! -d ./tmp ]; then
  mkdir tmp/
fi

echo "================== Feature Vectorization =================="
if [ ! -f ./tmp/dev2.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev2.tsv
  mv *.tsv tmp/
fi

python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 44247 \
./easynlp/appzoo/api.py \
--mode predict --tables tmp/dev2.tsv \
--input_schema label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
--learning_rate 5e-05 --epoch_num 3 --random_seed 1234 --predict_queue_size 1024 --predict_slice_size 4096 \
--predict_thread_num 1 --outputs tmp/dev.pred.tsv \
--output_schema pooler_output,first_token_output,all_hidden_outputs \
--append_cols label \
--sequence_length 128 --micro_batch_size 32 --app_name vectorization \
--first_sequence sent1 --checkpoint_dir bert-tiny-uncased
