accelerate launch --config_file config/sft.yaml train_sft.py \
    --data_path data/data.json \
    --model_path bigscience/bloom-1b1 \
    --save_path outputs/sft \
    --batch_size 16 \
    --max_length 384 \
    --epochs 4 \
    --lr 2e-5 \
    --weight_decay 0
