CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=4 cifar10_ddp.py >> ./nohup.log 2>&1 &
