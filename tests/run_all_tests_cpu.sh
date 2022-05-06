#!/usr/bin/env bash
conda create -n myenv python=3.6 

conda activate myenv

which python
python --version

set -e
export CUDA_VISIBLE_DEVICES=-1

# Cleanup environment
rm -rf $HOME/.easynlp_modelzoo
rm -rf $HOME/.cache/huggingface

export MKL_THREADING_LAYER=GNU

echo "================== Test user defined vectorization =================="
# check vectorization
sh run_vectorization.sh
