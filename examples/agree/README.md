# AGREE
This project is implemented for the WSDM2023 paper: "AGREE: Aligning Cross-Modal Entities for Image-Text Retrieval Upon Vision-Language Pre-trained Models". Our code is based on pytorch.

AGREE is a lightweight and practical approach to align cross-modal entities for image-text retrieval upon VLP models only at the fine-tuning and re-ranking stages. We employ external knowledge and tools to construct extra fine-grained image-text pairs, and then emphasize cross-modal entity alignment through contrastive learning and entity-level mask modeling in fine-tuning. Besides, two re-ranking strategies are proposed, including one specially designed for zero-shot scenarios.

We choose [CLIP](https://arxiv.org/abs/1806.08317) as the VLP model show our fine-tuning and re-ranking stages models in the repo.

## Data preparation

We use the popular public benchmark dataset Flickr30k for evaluation. You can download directly from the [official website](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html). We also provide the pre-processed data for convenience.

* Download the preprocessed training and validation [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/TACLDenotationGraph.pdf) data by:

    ```
    if [ ! -f ./tmp/datasets/flickr30k_images.tgz ]; then
        wget -P ./tmp/datasets https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/flick30k_images.tgz
        tar zxvf ./tmp/datasets/flickr30k_images.tgz -C ./tmp/datasets
    fi

    ```

*  The pre-processed training data contains our pre-extracted textual entities, and grounded textual entities with their predicted probabilities. The format of an item in " flickr30k_train_pred.jsonl " file is:

    ```
    {"image": "flickr30k-images/10002456.jpg", "caption": "Four men on top of a tall structure.", "image_id": 1, "segs": ["four men", "top of a tall structure", "top", "four men on top of a tall structure.", "a tall structure"], "preds": [{"men": 0.6230936527252198}, {"object": 0.6065886701856341}, {"top": 0.6251628398895264}, {"a tall structure": 0.5054895877838135}]}
    ```

We also provide the preprocessed and re-organized data of visual entities from Visual Genome, including image arrays and its mapping relationship with visual entities. Original data of Visual Genome comes from [URL](http://visualgenome.org/api/v0/api_home.html).

* Preprare the preprocessed data of visual entities by:

    ```
    if [ ! -f ./tmp/VG.tgz ]; then
        wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/VG.tgz
        tar zxvf ./tmp/VG.tgz -C ./tmp
    fi
    ```

Besides, we provide the images where the grounded textual entities are masked in the image. We utilize [GLIP](https://github.com/microsoft/GLIP) for visual grounding.

* Preprare the masked data of grounded textual entities by:

    ```
    if [ ! -f ./tmp/datasets/flickr30k_images/flickr30k_visual_grounding.224.npz ]; then
        wget -P ./tmp/datasets/flickr30k_images/ https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/flickr30k_visual_grounding.224.npz
    fi
    ```

## Model Preparation

* Download CLIP pretrained checkpoint (ViT-L/14 in this repo):

    ```
    if [ ! -f ./tmp/pretrained_models/ViT-L-14.pt ]; then
        wget -P ./tmp/pretrained_models https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easywxd/AGREE/openai/ViT-L-14.pt
    fi

    ```

## Fine-tuning

* Training

    ```
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
    ```


* Feature Extraction

  The resume path and experiment name should be replaced with your fine-tuned model.

    ```
    SAVEFOLDER=./tmp/predictions

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

    ```

  The output feature files are saved in user-defined $SAVEFOLDER in the following format:

  ("image_id" for images predictions and "query_id" for the texts)

  ```
  {"image_id": "9", "feature": [0.038518026471138,....]}
  {"query_id": 8, "feature": [-0.014490452595055103,....]}

  ```

* Predict

  Note that the prediction procedure should be executed after extracting features.

    ```
    python3 -u eval/make_topk_predictions.py \
        --image-feats="${SAVEFOLDER}/flickr30k_test_images.img_feat.jsonl" \
        --text-feats="${SAVEFOLDER}/flickr30k_test_texts.txt_feat.jsonl" \
        --top-k=10 \
        --eval-batch-size=32 \
        --output-images="${SAVEFOLDER}/test_images_predictions.jsonl" \
        --output-texts="${SAVEFOLDER}/test_texts_predictions.jsonl"

    ```
* Re-ranking

  For fine-tuning results, as the paper reported, we only utilize TBR (Text-Image Bi-directional Re-ranking) module for re-ranking.

    ```
    python3 -u eval/eval_back_retrieval.py "${DATAPATH}/annotation/flickr30k_test_images.jsonl" \
    "${SAVEFOLDER}/test_images_predictions.jsonl" \
    "${DATAPATH}/annotation/flickr30k_test_texts.jsonl" \
    "${SAVEFOLDER}/test_texts_predictions.jsonl" \
    "${SAVEFOLDER}/test_images_predictions_rerank.jsonl" \
    "${SAVEFOLDER}/test_texts_predictions_rerank.jsonl" \
    ```
  The commands will output prediction files after re-ranking.

* Evaluation

    ```
    python3 -u eval/evaluation.py \
    "${DATAPATH}/annotation/flickr30k_test_images.jsonl" \
    "${SAVEFOLDER}/test_images_predictions_rerank.jsonl" \
    "${SAVEFOLDER}/image_output_rerank.json" \
    image_id
    ```

  The evaluation procedure will read the predictions to compare with the ground-truth. Evaluation results are saved in user-defined "text_output_rerank.json" and "image_output_rerank.json" in the following format:

    ```
    {"success": true, "score": 95.93333333333334, "scoreJson": {"score": 95.93333333333334, "mean_recall": 95.93333333333334, "r1": 89.9, "r5": 98.4, "r10": 99.5}}
    ```

## Zero-shot Re-ranking

Here we provide examples on pre-trained CLIP (ViT-L/14) for zero-shot re-ranking with AGREE re-ranking procedures, including feature extraction, predict, re-ranking and evaluation.

* Feature Extraction

  The features of masked and prompted textual entities are also extracted, for re-ranking procedures.

    ```
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
    ```

  The extracted textual extities are pre-extracted in file " flickr30k_test_texts_segs.jsonl " for EGR (Textual Entity-Guided Re-ranking) module. 

* Predict

    ```
    python3 -u eval/make_topk_predictions_mask+prompt.py \
        --image-feats="${SAVEFOLDER}/flickr30k_test_imgs.224.img_feat.jsonl" \
        --text-feats="${SAVEFOLDER}/flickr30k_test_queries.txt_feat.jsonl" \
        --mask-feats="${SAVEFOLDER}/flickr30k_test_queries.mask_feat.jsonl" \
        --prompt-feats="${SAVEFOLDER}/flickr30k_test_queries.prompt_feat.jsonl" \
        --top-k=20 \
        --eval-batch-size=32 \
        --output-images="${SAVEFOLDER}/test_images_predictions_mask+prompt.jsonl" \
        --output-texts="${SAVEFOLDER}/test_texts_predictions_mask+prompt.jsonl"
    ```

* TBR Re-ranking and Evaluation

    ```
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
    ```


## Reference
* Learning Transferable Visual Models From Natural Language Supervision. [[paper](https://arxiv.org/abs/1806.08317)][[website](https://fashion-gen.com/)]
* From image descriptions to visual denotations:
New similarity metrics for semantic inference over event descriptions. [[paper](http://shannon.cs.illinois.edu/DenotationGraph/TACLDenotationGraph.pdf)][[website](http://shannon.cs.illinois.edu/DenotationGraph/data/index.html)]
* Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. [[paper](http://visualgenome.org/static/paper/Visual_Genome.pdf)][[website](http://visualgenome.org/)]
* Grounded Language-Image Pre-training. [[paper](https://arxiv.org/abs/2112.03857)][[website](https://github.com/microsoft/GLIP)]


## Acknowledgements

Our implementation of AGREE benefits from OpenAI's [CLIP](https://github.com/openai/CLIP) and the implementation version [OpenCLIP](https://github.com/mlfoundations/open_clip). The visual grounding parts are learnt from [GLIP](https://github.com/microsoft/GLIP). We thank the original authors for their open-sourcing. Thanks for their wonderful works.

