set -x
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
cur_path=/workspace/project_all/EasyNLP/examples/knowledge_distillation/metakd

# Download from https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D
model=~/.easynlp/modelzoo/tinybert
# You can also use our mirrored version
model=bert-tiny-uncased

# In domain_sentiment_data, genre is one of ["books", "dvd", "electronics", "kitchen"]
genre=books
EXEC=python
cd ${cur_path}

# 1. Distillation pretrain
DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
# Pretrained distillation
$EXEC -m torch.distributed.launch $DISTRIBUTED_ARGS meta_student_distill.py \
  --mode train \
  --tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
  --input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
  --first_sequence=text_a \
  --second_sequence=text_b \
  --label_name=label \
  --label_enumerate_values=positive,negative \
  --checkpoint_dir=./tmp/$genre/meta_student_pretrain/ \
  --learning_rate=3e-5  \
  --epoch_num=10  \
  --random_seed=42 \
  --logging_steps=20 \
  --sequence_length=128 \
  --micro_batch_size=16 \
  --app_name=text_classify \
  --user_defined_parameters="
                              pretrain_model_name_or_path=$model
                              teacher_model_path=./tmp/meta_teacher/
                              domain_loss_weight=0.5
                              distill_stage=first
                              genre=$genre
                              T=2
                            "

# 2. Finetune
pretrained_path="./tmp/$genre/meta_student_pretrain/"
$EXEC -m torch.distributed.launch $DISTRIBUTED_ARGS meta_student_distill.py \
  --mode train \
  --tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
  --input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
  --first_sequence=text_a \
  --second_sequence=text_b \
  --label_name=label \
  --label_enumerate_values=positive,negative \
  --checkpoint_dir=./tmp/$genre/meta_student_fintune/ \
  --learning_rate=3e-5  \
  --epoch_num=10  \
  --random_seed=42 \
  --logging_steps=20 \
  --save_checkpoint_steps=50 \
  --sequence_length=128 \
  --micro_batch_size=16 \
  --app_name=text_classify \
  --user_defined_parameters="
                              pretrain_model_name_or_path=$pretrained_path
                              teacher_model_path=./tmp/meta_teacher/
                              domain_loss_weight=0.5
                              distill_stage=second
                              genre=$genre
                              T=2
                            "

# 3. Evalute
Student_model_path=./tmp/$genre/meta_student_fintune/
$EXEC main_evaluate.py \
  --mode evaluate \
  --tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
  --input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
  --first_sequence=text_a \
  --label_name=label \
  --label_enumerate_values=positive,negative \
  --checkpoint_dir=./tmp/meta_teacher/ \
  --learning_rate=3e-5  \
  --epoch_num=1  \
  --random_seed=42 \
  --logging_steps=20 \
  --sequence_length=128 \
  --micro_batch_size=16 \
  --app_name=text_classify \
  --user_defined_parameters="pretrain_model_name_or_path=$Student_model_path
                              genre=$genre"
