# local training example
cur_path=/home/admin/workspace/EasyNLP/
cd ${cur_path}

rm -rf /home/admin/workspace/EasyNLP/examples/self_defined_examples/ckpt/*

# DDP multiple devices
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009 \
examples/self_defined_examples/main_mnist.py \
--mode train --tables "" --micro_batch_size=32 --worker_gpu=2 --worker_count=1 --worker_cpu=8

# DDP single device
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009 \
# examples/self_defined_examples/main_mnist.py \
# --mode train --tables "" --micro_batch_size=32 --worker_gpu=1 --worker_count=1 --worker_cpu=0
