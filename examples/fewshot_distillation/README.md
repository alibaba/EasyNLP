# Fewshot Distillation

## Overview

## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Prepare the data

You can download the datasets (SST-2, MR, CR, MNLI, SNLI, QNLI, RTE, QQP) [here](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/glue_extra.tgz). Please download it and extract the files to `./data`, or run the following commands:

```bash
mkdir data
cd data
bash download_dataset.sh
```

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python tools/generate_k_shot_data.py --mode k-shot --task sst-2 mr --k 16
python tools/generate_k_shot_data.py --mode k-shot --task sst-2 mr --k 160
```

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run

### Quick start
Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its `4.17.0` version. 

The pretrained models should be placed in folder 'pretrained' before starting training. You can download them from https://huggingface.co/models. You need to download at least two models **Roberta-large** and **Bert-small**.

```bash
mkdir pretrained
cd pretrained
ln -s ~/.../roberta-large-model-dir roberta-large
ln -s ~/.../bert-small-model-dir bert-small
```

We use cli.py as the program entry point and a series of python scripts to train the model.

## Train the model on dataset SST-2 and MR as example
**Train the in_domain Teacher and cross_domain Teacher**
```bash
python scripts/ptkd_teacher.py --device=0,1 -t 2 3 -k 16 -s 13
python scripts/ptkd_teacher.py --device=0,1 -t 2 3 -k 160 -s 13
```

**Compute domain expertise score**
```bash
python scripts/compute_weights_cls.py -t 2 3 -k 16 -s 13
python scripts/compute_weights_cls.py -t 2 3 -k 16 -s 13
```

**Train the corss_domain Student using distillation and domain expertise score**
```bash
python scripts/ptkd_student_weights_man.py --device=0,1 -t 2 3 -k 160 -s 13
```
**Train the in_domain Student with the help of corss_domain Student**
```bash
python scripts/cptkd_student_weights_man.py --device=0,1 -t 2 3 -k 16 -s 13
```
The implementation of this code is based on [LM-BFF](https://github.com/princeton-nlp/LM-BFF).