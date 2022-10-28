#export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

mode=inter
#mode=intra
N=5 # 5, 5, 10, 10
Q=1 # 1, 5, 1, 1
K=1 # 1, 5, 1, 5

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=6012 nlp_trainer.py \
  --model_name_or_path=/wjn/pre-trained-lm/bert-base-uncased \
  --data_dir=/wjn/projects/few-shot_ner/SpanProto/datasets/episode-data/$mode \
  --output_dir=./outputs/"$mode-$N-$K" \
  --seed=42 \
  --exp_name=few-nerd \
  --max_seq_length=64 \
  --max_eval_seq_length=64 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --learning_rate=2e-05 \
  --num_train_epochs=3 \
  --logging_steps=100000000 \
  --eval_steps=1000 \
  --save_steps=1000 \
  --save_total_limit=1 \
  --warmup_steps=1000 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=fewnerd \
  --task_type=span_proto \
  --model_type=bert \
  --metric_for_best_model=class_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=short_labels \
  --keep_predict_labels \
  --dataloader_num_workers 0 \
  --user_defined="N=$N Q=$Q K=$K mode=$mode"
#  --do_adv