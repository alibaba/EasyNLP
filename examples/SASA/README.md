# Understanding Long Programming Languages with Structure-Aware Sparse Attention

## 准备数据和模型

以Defect Detection任务为例，下载数据：

```
if [ ! -f ./tmp/long_valid.jsonl ]; then
wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/long_train.jsonl
wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/long_valid.jsonl
wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/long_test.jsonl
fi
```

加载预处理的attention matrix

```
if [ ! -f ./tmp/topk_ast_count.pt ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/topk_ast_count.pt
fi
```

加载codebert-1024模型ckpt

```
if [ ! -f ./tmp/codebert-base-1024.tgz ]; then
wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/codebert-base-1024.tgz
tar zxvf ./tmp/codebert-base-1024.tgz -C ./tmp
fi
```



## 开始训练

### 训练脚本

```
OUTPUT_DIR=./tmp/SASA_finetune
PRETRAINED_MODEL=./tmp/codebert-base-1024/

python main.py 
--output_dir=$OUTPUT_DIR 
--tokenizer_name=$PRETRAINED_MODEL 
--model_name_or_path=$PRETRAINED_MODEL 
--model_type=topk_ast 
--do_train 
--train_data_file=./tmp/long_train.jsonl 
--eval_data_file=./tmp/long_valid.jsonl 
--epoch 5 
--seq_len 1024 
--block_size 32 
--num_random_blocks 3 
--train_batch_size 8 
--eval_batch_size 32 
--learning_rate 2e-5 
--max_grad_norm 1.0 
--evaluate_during_training 
--seed 123456
```



### 测试脚本

```
OUTPUT_DIR=./tmp/SASA_finetune
PRETRAINED_MODEL=./tmp/codebert-base-1024/

python main.py \
    --output_dir=$OUTPUT_DIR \
    --tokenizer_name=$PRETRAINED_MODEL \
    --model_name_or_path=$PRETRAINED_MODEL \
    --model_type=topk_ast \
    --do_test \
    --train_data_file=./tmp/long_train.jsonl \
    --eval_data_file=./tmp/long_valid.jsonl \
    --test_data_file=./tmp/long_test.jsonl \
    --seq_len 1024 \
    --block_size 32 \
    --num_random_blocks 3 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456
```

### 性能评估

```
python evaluator.py -a ./tmp/long_test.jsonl -p ${OUTPUT_DIR}/predictions.txt
```

## 相关论文

Tingting Liu, Chengyu Wang, Cen Chen, Ming Gao, Aoying Zhou. Understanding Long Programming Languages with Semantics-Aware Sparse Attention. SIGIR 2022

```
@inproceedings{sigir2022,
    author    = {Tingting Liu and
                Chengyu Wang and
                Cen Chen and
                Ming Gao and
                Aoying Zhou},
    title     = {Understanding Long Programming Languages with Structure-Aware Sparse Attention},
    booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages     = {2093–-2098},
    year      = {2022}
  }
```





