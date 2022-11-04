#export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3


ID=4 # 1, 2, 3, 4
K=5 # 1, 5

if [ "$ID" = "1" ]; then
  N=4

elif [ "$ID" = "2" ]; then
  N=11

elif [ "$ID" = "3" ]; then
  N=6

elif [ "$ID" = "4" ]; then
  N=18

fi

if [ "$K" = "1" ]; then
  mode=xval_ner

elif [ "$K" = "5" ]; then
  mode=xval_ner_shot_5

fi

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=6044 nlp_trainer.py \
  --model_name_or_path=/wjn/pre-trained-lm/bert-base-uncased \
  --data_dir=/wjn/fewshot/SpanProto/dataset \
  --output_dir=./outputs/"$mode-$ID" \
  --seed=42 \
  --exp_name=cross-ner \
  --max_seq_length=64 \
  --max_eval_seq_length=64 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --evaluation_strategy=steps \
  --learning_rate=2e-05 \
  --num_train_epochs=100 \
  --logging_steps=100000000 \
  --eval_steps=500 \
  --save_steps=500 \
  --save_total_limit=1 \
  --warmup_steps=500 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=crossner \
  --task_type=span_proto \
  --model_type=bert \
  --metric_for_best_model=class_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=short_labels \
  --keep_predict_labels \
  --dataloader_num_workers 1 \
  --user_defined="N=$N ID=$ID K=$K mode=$mode"
#  --do_adv