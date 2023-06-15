#!/bin/sh

CUDA_VISIBLE_DEVICES=5 python train.py \
--csv_path data/clean_data_ori/english_train.csv \
--model_save_path ./result/  \
--model_name test_t5  \
--train_epoch 10 \
--model_for_train t5-large \
--val_json_path data/clean_data_ori/english_val.json \
--eval_json_path data/clean_data_ori/english_eval.json \
--batch_size 1 \
--config_path models/test_config/t5_large_config.json \
--lr 3e-5 \
> ./result/txt/test_t5.txt