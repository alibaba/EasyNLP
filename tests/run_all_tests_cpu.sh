#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES=-1

# Cleanup environment
rm -rf $HOME/.easynlp_modelzoo
rm -rf $HOME/.cache/huggingface

export MKL_THREADING_LAYER=GNU

# Install easynlp cli
cd ../
pip uninstall easynlp -y
python setup.py install --user
cd tests/

if [ ! -d ./tmp ]; then
  mkdir tmp/
fi

rm -rf *.tsv *.csv *.txt


# echo "================== Test user defined vectorization =================="
# check vectorization
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

echo "================== Test TorchACC =================="
python test_torchacc.py
rm -rf *.tsv

# echo "================== Feature Vectorization =================="
# if [ ! -f ./dev2.tsv ]; then
#  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev2.tsv
# fi

# python test_vectorization.py
# rm -rf *.tsv


# Unit tests
echo "================== Test user defined example =================="
if [ ! -f ./train.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv
fi

python test_classification_self_defined.py
rm -rf *.tsv


echo "================== Test language modeling =================="
if [ ! -f ./train.json ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/train.json
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/language_modeling/dev.json
fi

python test_language_modeling.py
rm -rf tmp/*
rm -rf *.json *.csv *.tsv

echo "================== Test data augmentation =================="
python test_data_augmentation.py
rm -rf *.tsv

# # echo "================== Test few shot =================="
# # python test_few_shot.py
# # rm -rf few_shot_model few_shot_logs -f
# # rm -rf *.tsv -f

#
# uninstall easynlp
#
pip uninstall easynlp -y

