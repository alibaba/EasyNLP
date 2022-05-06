#!/usr/bin/env bash

set -e
export CUDA_VISIBLE_DEVICES=-1

# Cleanup environment
rm -rf $HOME/.easynlp_modelzoo
rm -rf $HOME/.cache/huggingface

export MKL_THREADING_LAYER=GNU

echo "================== Test user defined vectorization =================="
# check vectorization
sh run_vectorization.sh

# Install easynlp cli
cd ../
pip uninstall easynlp -y
python setup.py install --user
cd tests/

if [ ! -d ./tmp ]; then
  mkdir tmp/
fi

rm -rf *.tsv *.csv *.txt
