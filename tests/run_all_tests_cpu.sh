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

echo "================== Test TorchACC =================="
python test_torchacc.py
rm -rf *.tsv

echo "================== Feature Vectorization =================="
if [ ! -f ./dev2.tsv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev2.tsv
fi

python test_vectorization.py
rm -rf *.tsv
