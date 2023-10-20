accelerate launch --config_file config/ppo.yaml train_ppo.py \
    --data_path data/data.json \
    --model_path outputs/sft \
    --aes_model_path outputs/aes \
    --ps_model_path outputs/ps \
    --save_path outputs/ppo \
    --num_layers_unfrozen 8 \
    --total_steps 5000 \
    --alpha 0.7 \
    --batch_size 4
