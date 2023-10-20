GPU=4,5,6,7

REPO=$PWD

DATA_DIR=$REPO/data
MODEL_NAME_OR_PATH='bert-base-multilingual-cased'

n_gpu=4
epoch=1
bsz=8
grad_acc=1
wd=0.0001

lr=3e-5

alpha=0.2
mix_layer=7

OUTPUT_DIR=$REPO/outputs/mbert_xquad
mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port=$RANDOM ./run_xqa.py \
    --task_name xquad \
    --data_dir $DATA_DIR \
    --model_type bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --language en,ar,de,el,es,hi,ru,th,tr,vi,zh \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size $bsz \
    --gradient_accumulation_steps $grad_acc \
    --learning_rate ${lr} \
    --per_gpu_eval_batch_size 32 \
    --num_train_epochs $epoch \
    --eval_steps 500 \
    --max_seq_length 384 \
    --doc_stride 128  \
    --output_dir $OUTPUT_DIR \
    --log_dir $OUTPUT_DIR \
    --threads 8 \
    --cache_dir $DATA_DIR/caches_mbert_mlqa \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay $wd \
    --consist_weight 0.05 \
    --teaching_weight 0.1 \
    --align_weight 0.05 \
    --alpha $alpha \
    --mix_layer $mix_layer \
    --norm \
    --cl

