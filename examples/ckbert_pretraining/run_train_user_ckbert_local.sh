export PYTHONPATH=$PYTHONPATH:/mnt/djw/latest/EasyNLP
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

gpu_number=1
negative_e_number=4
negative_e_length=16

base_dir=$PWD
checkpoint_dir=$base_dir/checkpoints
resources=$base_dir/resources
local_kg=$resources/ownthink_triples_small.txt
local_train_file=$resources/train_small.txt
remote_kg=https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ckbert/ownthink_triples_small.txt
remote_train_file=https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/ckbert/train_small.txt

if [ ! -d $checkpoint_dir ];then
    mkdir $checkpoint_dir
fi

if [ ! -d $resources ];then
    mkdir $resources
fi

if [ ! -f $local_kg ];then
    wget -P $resources $remote_kg
fi

if [ ! -f $local_train_file ];then
    wget -P $resources $remote_train_file
fi

python -m torch.distributed.launch --nproc_per_node=$gpu_number \
--master_port=52349 \
$base_dir/main.py \
--mode=train \
--worker_gpu=$gpu_number \
--tables=$local_train_file, \
--learning_rate=5e-5  \
--epoch_num=5  \
--logging_steps=10 \
--save_checkpoint_steps=2150 \
--sequence_length=256 \
--train_batch_size=20 \
--checkpoint_dir=$checkpoint_dir \
--app_name=language_modeling \
--use_amp \
--save_all_checkpoints \
--user_defined_parameters="pretrain_model_name_or_path=hfl/macbert-base-zh external_mask_flag=True contrast_learning_flag=True negative_e_number=${negative_e_number} negative_e_length=${negative_e_length} kg_path=${local_kg}"