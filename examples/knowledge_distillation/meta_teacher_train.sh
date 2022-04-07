set -x
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

EXEC='python'

cur_path=/workspace/project_all/EasyNLP/examples/knowledge_distillation/metakd
model=bert-base-uncased
cd ${cur_path}

DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
$EXEC -m torch.distributed.launch $DISTRIBUTED_ARGS meta_teacher_train.py \
  --mode train \
  --tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
  --input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
  --first_sequence=text_a \
  --second_sequence=text_b \
  --label_name=label \
  --label_enumerate_values=positive,negative \
  --checkpoint_dir=./tmp/meta_teacher/ \
  --learning_rate=3e-5  \
  --epoch_num=1  \
  --random_seed=42 \
  --logging_steps=20 \
  --save_checkpoint_steps=50 \
  --sequence_length=128 \
  --micro_batch_size=16 \
  --app_name=text_classify \
  --user_defined_parameters="
                              pretrain_model_name_or_path=$model
                              use_sample_weights=True
                              use_domain_loss=True
                              domain_loss_weight=0.5
                              "
