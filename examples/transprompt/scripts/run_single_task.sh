python3 cli.py \
        --task_type single_task \
        --task_name mrpc \
        --k 16 \
        --seed 42 \
        --data_dir data/k-shot-single/MRPC/16-42 \
        --model_type roberta \
        --model_name_or_path roberta-large \
        --output_dir output \
        --do_eval \
        --do_train \
        --embed_size 1024 \
        --pet_per_gpu_eval_batch_size 8 \
        --pet_per_gpu_train_batch_size 8 \
        --pet_max_seq_length 128 \
        --pet_max_steps 8000 \
        --warmup_steps 100 \
        --eval_every_step 400 \
        --learning_rate 1e-5 \
        --overwrite_output_dir
