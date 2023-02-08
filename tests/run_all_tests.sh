#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

# Cleanup environment
rm -rf $HOME/.easynlp_modelzoo
rm -rf $HOME/.cache/huggingface

export MKL_THREADING_LAYER=GNU

# # check vectorization
# sh run_vectorization.sh

# Install easynlp cli
cd ../
pip uninstall easynlp -y
python setup.py install --user
cd tests/

if [ ! -d ./tmp ]; then
  mkdir tmp/
fi

rm -rf *.tsv *.csv *.txt

#echo "================== Test MegatronBERT classification =================="
#if [ ! -f ./train.tsv ]; then
#  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
#  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
#fi

#python test_megatron_bert.py
#rm -rf classification_model

echo "================== Test DKPLM =================="
if [ ! -f ./train_corpus.txt ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/train_corpus.txt
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/dev_corpus.txt
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/entity_emb.txt
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dkplm/rel_emb.txt
fi
python test_dkplm_language_modeling.py
rm -rf *.txt

#echo "================== Test TorchACC =================="
#python test_torchacc.py
#rm -rf *.tsv

# echo "================== Feature Vectorization =================="
# if [ ! -f ./dev2.tsv ]; then
#  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev2.tsv
# fi

# python test_vectorization.py
# rm -rf *.tsv

echo "================== Test classification =================="
if [ ! -f ./train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

python test_classification.py
rm -rf classification_model

echo "================== Test geep classification =================="
if [ ! -f ./train_toy.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train_toy.tsv
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

python test_geep_classification.py
rm -rf classification_model

echo "================== Test clip =================="
if [ ! -f ./MUGE_MR_train_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_train_base64_part.tsv
fi

if [ ! -f ./MUGE_MR_valid_base64_part.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_valid_base64_part.tsv
fi

if [ ! -f ./MUGE_MR_test_base64_part_text.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_test_base64_part_text.tsv
fi

python test_clip.py
rm -rf modelzoo_alibaba.json
rm -rf clip_cn_model
rm -rf *.tsv

echo "================== Test user defined example =================="
python test_classification_self_defined.py
rm -rf *.tsv

echo "================== Test text match single/two tower=================="
if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/tutorials/ez_text_match/afqmc_public/train.csv
fi
if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/tutorials/ez_text_match/afqmc_public/dev.csv
fi
python test_text_match_single_tower.py
rm -rf text_match_single_tower_model_dir
# python test_text_match_two_tower.py
# rm -rf text_match_two_tower_model_dir
rm -rf *.csv

echo "================== Test sequence labeling =================="

if [ ! -f ./test.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/train.csv
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/dev.csv
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sequence_labeling/test.csv
fi

python test_sequence_labeling.py
rm -rf labeling_model
rm -rf *.txt *.csv *.tsv

echo "================== Test language modeling =================="
if [ ! -f ./train.json ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/train.json
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dev.json
fi

python test_language_modeling.py
rm -rf tmp/*
rm -rf *.json *.csv *.tsv

# echo "================== Test classification w. knowledge distillation =================="
# python test_classification_distillation.py
# rm -rf classification_model
# rm -rf *.txt *.csv *.tsv
# rm -rf results

echo "================== Test data augmentation =================="
python test_data_augmentation.py
rm -rf *.tsv

echo "================== Test few shot =================="
python test_few_shot.py
rm -rf few_shot_model few_shot_logs -f
rm -rf *.tsv -f

echo "================== Test sequence generation =================="
if [ ! -f ./cn_train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easytexminer/tutorials/generation/cn_train.tsv
fi

if [ ! -f ./cn_dev.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easytexminer/tutorials/generation/cn_dev.tsv
fi

python test_sequence_generation.py

rm -rf sequence_generation_model
rm -rf *.txt *.csv *.tsv

echo "================== Test machine reading comprehension =================="
if [ ! -f ./train_squad.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/train_squad.tsv
fi

if [ ! -f ./dev_squad.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/machine_reading_comprehension/dev_squad.tsv
fi

python test_machine_reading_comprehension.py

rm -rf mrc_model
rm -rf *.txt *.csv *.tsv

# uninstall easynlp

pip uninstall easynlp -y
