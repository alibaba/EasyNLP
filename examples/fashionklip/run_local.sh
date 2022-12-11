export PYTHONPATH="$PYTHONPATH:$PWD/src"
export CUDA_VISIBLE_DEVICES=$1

MASTER_ADDR=tcp://127.0.0.1:11907

mode=$2

if [ ! -f ./tmp/fashion-gen.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/fashion-gen.tgz
    tar zxvf ./tmp/fashion-gen.tgz -C ./tmp
fi

if [ ! -f ./tmp/fashion_kb.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/fashion_kb.tgz
    tar zxvf ./tmp/fashion_kb.tgz -C ./tmp
fi

if [ ! -f ./tmp/pretrained_models/ViT-B-32.pt ]; then
    wget -P ./tmp2/pretrained_models https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/openai/ViT-B-32.pt
fi

if [ ! -f ./tmp/pretrained_models/pai-clip-commercial-base-en.tgz ]; then
    wget -P ./tmp/pretrained_models https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/pai-clip-commercial-base-en.tgz
    tar zxvf ./tmp/pretrained_models/pai-clip-commercial-base-en.tgz -C ./tmp/pretrained_models
fi

DATAPATH=./tmp
PRETRAINED_MODEL=./tmp/pretrained_models
SAVEFOLDER=./tmp/predictions_finetune

if [ "$mode" = "finetune" ]; then

    python3 -u training/main_all_concept.py \
        --save-most-recent \
        --save-frequency 1 \
        --report-to tensorboard \
        --train-data="${DATAPATH}/fashiongen/train/fashion-gen_train_queries_phrases.jsonl"  \
        --train-img="${DATAPATH}/fashiongen/train/train.224.npz" \
        --txt-id-filename="${DATAPATH}/fashiongen/train/fashion-gen_concepts_queries.jsonl" \
        --kb-txt-id-filename="${DATAPATH}/fashion_kb/icbu_concepts_queries.jsonl" \
        --val-data="${DATAPATH}/fashiongen/val/fashion-gen_val_queries.jsonl"  \
        --val-img="${DATAPATH}/fashiongen/val/val.224.npz" \
        --img-data-sets="${DATAPATH}/fashion_kb/icbu_train_images_5w_00.224.npz" \
        --concept-data="${DATAPATH}/fashiongen/train/fashion-gen_concepts_queries.jsonl" \
        --kb-concept-data="${DATAPATH}/fashion_kb/icbu_concepts_queries.jsonl" \
        --resume="${PRETRAINED_MODEL}/pai-clip-commercial-base-en/pai-clip-commercial-base-en.pt" \
        --is-concept \
        --is-data-concept \
        --is-update \
        --dist-url=$MASTER_ADDR \
        --dataset-type jsonl \
        --warmup 500 \
        --batch-size=32 \
        --eval-batch-size=256 \
        --lr=1e-5 \
        --wd=0.001 \
        --epochs=25 \
        --workers=0 \
        --model ViT-B/32

elif [ "$mode" = "finetune_only" ]; then

    python3 -u training/main_all_concept.py \
        --save-most-recent \
        --save-frequency 1 \
        --report-to tensorboard \
        --train-data="${DATAPATH}/fashiongen/train/fashion-gen_train_queries_phrases.jsonl"  \
        --train-img="${DATAPATH}/fashiongen/train/train.224.npz" \
        --txt-id-filename="${DATAPATH}/fashiongen/train/fashion-gen_concepts_queries.jsonl" \
        --kb-txt-id-filename="${DATAPATH}/fashion_kb/icbu_concepts_queries.jsonl" \
        --val-data="${DATAPATH}/fashiongen/val/fashion-gen_val_queries.jsonl"  \
        --val-img="${DATAPATH}/fashiongen/val/val.224.npz" \
        --img-data-sets="${DATAPATH}/fashion_kb/icbu_train_images_5w_00.224.npz" \
        --concept-data="${DATAPATH}/fashiongen/train/fashion-gen_concepts_queries.jsonl" \
        --kb-concept-data="${DATAPATH}/fashion_kb/icbu_concepts_queries.jsonl" \
        --is-concept \
        --is-data-concept \
        --is-update \
        --dist-url=$MASTER_ADDR \
        --dataset-type jsonl \
        --warmup 500 \
        --batch-size=32 \
        --eval-batch-size=256 \
        --lr=1e-5 \
        --wd=0.001 \
        --epochs=30 \
        --workers=0 \
        --model ViT-B/32 \
        --openai-pretrained

elif [ "$mode" = "evaluate" ]; then

    if [ ! -d $SAVEFOLDER ]; then
        mkdir $SAVEFOLDER
    fi

    python3 -u eval/extract_features.py \
        --extract-image-feats \
        --extract-text-feats \
        --image-data="${DATAPATH}/fashiongen/val/val.224.npz" \
        --text-data="${DATAPATH}/fashiongen/val/fashion-gen_val_queries.jsonl" \
        --img-batch-size=32 \
        --text-batch-size=32 \
        --resume="${PRETRAINED_MODEL}/pai-clip-commercial-base-en/pai-clip-commercial-base-en.pt" \
        --image-feat-output-path="${SAVEFOLDER}/fashion-gen_test_imgs.img_feat.jsonl" \
        --text-feat-output-path="${SAVEFOLDER}/fashion-gen_test_texts.txt_feat.jsonl" \
        --model ViT-B-32

    python3 -u eval/predict_evaluate.py \
        --image-feats="${SAVEFOLDER}/fashion-gen_test_imgs.img_feat.jsonl" \
        --text-feats="${SAVEFOLDER}/fashion-gen_test_texts.txt_feat.jsonl" \
        --top-k=10 \
        --eval-batch-size=32768 \
        --output-images="${SAVEFOLDER}/test_imgs_predictions.jsonl" \
        --output-texts="${SAVEFOLDER}/test_txts_predictions.jsonl" \
        --text-standard-path="${DATAPATH}/fashiongen/val/fashion-gen_val_queries.jsonl" \
        --image-standard-path="${DATAPATH}/fashiongen/val/fashion-gen_val_images.jsonl" \
        --text-out-path="${SAVEFOLDER}//fashion_text_output.json" \
        --image-out-path="${SAVEFOLDER}/./fashion_image_output.json"

fi

# --resume="${PRETRAINED_MODEL}/pai-clip-commercial-base-en/pai-clip-commercial-base-en.pt" \