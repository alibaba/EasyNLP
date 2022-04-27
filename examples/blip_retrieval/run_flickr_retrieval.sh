export CUDA_VISIBLE_DEVICES=0

OUTPUT=output/finetune_flickr

GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE"

python -m torch.distributed.launch $DISTRIBUTED_ARGS train_retrieval.py \
    --output_dir $OUTPUT \
    --evaluate \
    --image_root='../../../../../datasets/flickr30k_images/' \
    --ann_root='../../../../../datasets/flickr30k_images/annotation/' \
    --dataset='flickr' \
    --train_file='flickr30k_train.json' \
    --val_file='flickr30k_val.json' \
    --test_file='flickr30k_test.json' \
    --pretrained='http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/blip/pretrained/model_base_retrieval_flickr.pth' \
    --vit='base' \
    --batch_size_train=32 \
    --batch_size_test=64 \
    --vit_grad_ckpt=True \
    --vit_ckpt_layer=4 \
    --init_lr=1e-5 \
    --image_size=384 \
    --queue_size=57600 \
    --alpha=0.4 \
    --k_test=128 \
    --negative_all_rank=False \
    --weight_decay=0.05 \
    --min_lr=0 \
    --max_epoch=6
