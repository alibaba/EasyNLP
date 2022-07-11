export CUDA_VISIBLE_DEVICES=$1
mode=$2

if [ ! -f ./tmp/long_valid.jsonl ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/long_train.jsonl
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/long_valid.jsonl
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/long_test.jsonl
fi

if [ ! -f ./tmp/codebert-base-1024.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/codebert-base-1024.tgz
    tar zxvf ./tmp/codebert-base-1024.tgz -C ./tmp
fi

if [ ! -f ./tmp/topk_ast_count.pt ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/SASA/topk_ast_count.pt
fi

if [ "$mode" = "train" ]; then
OUTPUT_DIR=./tmp/SASA_finetune
PRETRAINED_MODEL=./tmp/codebert-base-1024/

python main.py \
    --output_dir=$OUTPUT_DIR \
    --tokenizer_name=$PRETRAINED_MODEL \
    --model_name_or_path=$PRETRAINED_MODEL \
    --model_type=topk_ast \
    --do_train \
    --train_data_file=./tmp/long_train.jsonl \
    --eval_data_file=./tmp/long_valid.jsonl \
    --epoch 5 \
    --seq_len 1024 \
    --block_size 32 \
    --num_random_blocks 3 \
    --train_batch_size 8 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 


elif [ "$mode" = "predict" ]; then
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

python evaluator.py -a ./tmp/long_test.jsonl -p ${OUTPUT_DIR}/predictions.txt
fi