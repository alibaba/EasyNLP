accelerate launch --config_file config/sft.yaml train_rm.py \
    --data_path data/data.json \
    --save_path outputs/rm_ps \
    --batch_size 16 \
    --max_length 384 \
    --model_path bigscience/bloom-1b1 \
    --epochs 1 \
    --lr 1e-5 \
    --rm_type ps
