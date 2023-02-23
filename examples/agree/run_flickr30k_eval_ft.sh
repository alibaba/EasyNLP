export PYTHONPATH="$PYTHONPATH:$PWD/src"
# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0

DATAPATH=./tmp/datasets/flickr30k_images
experiment_name="./logs/lr=1e-05_wd=0.1_agg=True_model=ViT-L/14_batchsize=8_workers=0_date=2023-02-22-03-44-02"
SAVEFOLDER="./tmp/predictions/flickr30k/finetune"

if [ ! -d $SAVEFOLDER ]; then
        mkdir $SAVEFOLDER
fi

echo "=========== Feature Extraction ==========="

python3 -u eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/annotation/flickr30k_test_images.npz" \
    --text-data="${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --resume="${experiment_name}/checkpoints/epoch_2.pt" \
    --image-feat-output-path="${SAVEFOLDER}/flickr30k_test_images.img_feat.jsonl" \
    --text-feat-output-path="${SAVEFOLDER}/flickr30k_test_texts.txt_feat.jsonl" \
    --model ViT-L-14

echo "=========== Predict ==========="

python3 -u eval/make_topk_predictions.py \
    --image-feats="${SAVEFOLDER}/flickr30k_test_images.img_feat.jsonl" \
    --text-feats="${SAVEFOLDER}/flickr30k_test_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32 \
    --output-images="${SAVEFOLDER}/test_images_predictions.jsonl" \
    --output-texts="${SAVEFOLDER}/test_texts_predictions.jsonl"

echo "=========== TBR Re-ranking ==========="

python3 -u eval/eval_back_retrieval.py "${DATAPATH}/annotation/flickr30k_test_images.jsonl" \
"${SAVEFOLDER}/test_images_predictions.jsonl" \
"${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
"${SAVEFOLDER}/test_texts_predictions.jsonl" \
"${SAVEFOLDER}/test_images_predictions_rerank.jsonl" \
"${SAVEFOLDER}/test_texts_predictions_rerank.jsonl" \


echo "=========== Evaluation ==========="

python3 -u eval/evaluation.py \
"${DATAPATH}/annotation/flickr30k_test_images.jsonl" \
"${SAVEFOLDER}/test_images_predictions_rerank.jsonl" \
"${SAVEFOLDER}/image_output_rerank.json" \
image_id

python3 -u eval/evaluation.py \
"${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
"${SAVEFOLDER}/test_texts_predictions_rerank.jsonl" \
"${SAVEFOLDER}/text_output_rerank.json" \
query_id
