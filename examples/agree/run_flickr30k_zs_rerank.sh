export PYTHONPATH="$PYTHONPATH:$PWD/src"
# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0

DATAPATH=./tmp/datasets/flickr30k_images
MODELPATH=./tmp/pretrained_models
SAVEFOLDER="./tmp/predictions/flickr30k/zero-shot"

if [ ! -d $SAVEFOLDER ]; then
        mkdir $SAVEFOLDER
fi

echo "=========== Feature Extraction ==========="

python3 -u eval/extract_features_zs_mask+prompt.py \
    --extract-image-feats \
    --extract-text-feats \
    --extract-mask-feats \
    --extract-prompt-feats \
    --image-data="${DATAPATH}/annotation/flickr30k_test_images.npz" \
    --text-data="${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
    --prompt-data="${DATAPATH}/annotation/flickr30k_test_texts_segs.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --resume="${MODELPATH}/ViT-L-14.pt" \
    --image-feat-output-path="${SAVEFOLDER}/flickr30k_test_imgs.224.img_feat.jsonl" \
    --text-feat-output-path="${SAVEFOLDER}/flickr30k_test_queries.txt_feat.jsonl" \
    --mask-feat-output-path="${SAVEFOLDER}/flickr30k_test_queries.mask_feat.jsonl" \
    --prompt-feat-output-path="${SAVEFOLDER}/flickr30k_test_queries.prompt_feat.jsonl" \
    --model ViT-L-14

echo "=========== Predict + Prompt + Mask Re-ranking ==========="

python3 -u eval/make_topk_predictions_mask+prompt.py \
    --image-feats="${SAVEFOLDER}/flickr30k_test_imgs.224.img_feat.jsonl" \
    --text-feats="${SAVEFOLDER}/flickr30k_test_queries.txt_feat.jsonl" \
    --mask-feats="${SAVEFOLDER}/flickr30k_test_queries.mask_feat.jsonl" \
    --prompt-feats="${SAVEFOLDER}/flickr30k_test_queries.prompt_feat.jsonl" \
    --top-k=20 \
    --eval-batch-size=32 \
    --output-images="${SAVEFOLDER}/test_images_predictions_mask+prompt.jsonl" \
    --output-texts="${SAVEFOLDER}/test_texts_predictions_mask+prompt.jsonl"

echo "=========== TBR Re-ranking ==========="

python3 -u eval/eval_back_retrieval.py "${DATAPATH}/annotation/flickr30k_test_images.jsonl" \
"${SAVEFOLDER}/test_images_predictions_mask+prompt.jsonl" \
"${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
"${SAVEFOLDER}/test_texts_predictions_mask+prompt.jsonl" \
"${SAVEFOLDER}/test_images_predictions_mask+prompt_rerank.jsonl" \
"${SAVEFOLDER}/test_texts_predictions_mask+prompt_rerank.jsonl" \

echo "=========== Evaluation ==========="

python3 -u eval/evaluation.py "${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
"${SAVEFOLDER}/test_texts_predictions_mask+prompt_rerank.jsonl" \
"${SAVEFOLDER}/text_output_mask+prompt_rerank.json" \
query_id

python3 -u eval/evaluation.py "${DATAPATH}/annotation/flickr30k_test_images.jsonl" \
"${SAVEFOLDER}/test_images_predictions_mask+prompt_rerank.jsonl" \
"${SAVEFOLDER}/image_output_mask+prompt_rerank.json" \
image_id
