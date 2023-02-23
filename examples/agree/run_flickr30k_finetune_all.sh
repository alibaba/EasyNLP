export PYTHONPATH="$PYTHONPATH:$PWD/src"
export CUDA_VISIBLE_DEVICES=1,2,3,4

DATAPATH=./tmp/datasets/flickr30k_images

python3 -u training/main_all.py \
    --save-frequency 2 \
    --report-to tensorboard \
    --train-data="${DATAPATH}/annotation/flickr30k_train_pred.jsonl"  \
    --val-data="${DATAPATH}/annotation/flickr30k_val.json"  \
    --img-key image \
    --caption-key caption \
    --dataset-type json \
    --is-mask \
    --is-prompt \
    --is-da-loss \
    --is-da-mask \
    --is-vg \
    --dist-url="tcp://127.0.0.1:13526" \
    --warmup 10000 \
    --batch-size=8 \
    --lr=1e-5 \
    --wd=0.1 \
    --epochs=50 \
    --workers=0 \
    --model ViT-L/14 \
    --openai-pretrained