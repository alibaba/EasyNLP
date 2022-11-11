export CUDA_VISIBLE_DEVICES=$1

MASTER_ADDR=localhost
MASTER_PORT=6027
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

if [ ! -f ./MUGE_MR_test_base64_part_text.tsv ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/CLIP/MUGE_MR_test_base64_part_text.tsv
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
mode=$2

if [ "$mode" = "predict" ]; then
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
      --mode predict \
      --worker_gpu=1 \
      --tables=./MUGE_MR_test_base64_part_text.tsv \
      --input_schema=text:str:1 \
      --output_schema=text \
      --outputs ./output_placeholder.tsv \
      --first_sequence=text \
      --checkpoint_dir=alibaba-pai/latent_diffusion_fashion_cn_860M \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=latent_diffusion \
      --user_defined_parameters="n_samples=2 write_image=True image_prefix=./output/" 
fi
