# FashionKLIP

## Preparations: Data and Models

* Download the preprocessed training and validation [FashionGen](https://arxiv.org/abs/1806.08317) data:

```
if [ ! -f ./tmp/fashion-gen.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/fashion-gen.tgz
    tar zxvf ./tmp/fashion-gen.tgz -C ./tmp
fi
```

* Preprare our FashionMMKG sample data:

```
if [ ! -f ./tmp/fashion_kb.tgz ]; then
    wget -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/fashion_kb.tgz
    tar zxvf ./tmp/fashion_kb.tgz -C ./tmp
fi
```

* For confidentiality reasons, we only provide sample data of FashionMMKG. If you'd like to construct a larger MMKG or utilize the data of your own for better knowledge injected training, we suggest that the knowledge data organized in the following schema. We take the processed concept files in ./tmp/fashion_kb as examples.

  * concepts_images_sample.224.npz: image file in npz format, including image ids with its numpy array after transformation.
  * concepts_fathers.jsonl: "phrase" for specific concept, "phrase_father" for concept hypernym.
  ```
  {"query_id": 15, "phrase": "woman sunglasses", "phrase_father": "sunglasses"}
  ```
  * concepts_queries.jsonl: "phrase" for specific concept, "query_text" for concept prompt.
  ```
  {"query_id": 15, "query_text": "a photo of woman sunglasses", "phrase": "woman sunglasses"}
  ```

* Download CLIP pretrained checkpoint (ViT-B/32 as image encoder):

```
if [ ! -f ./tmp/pretrained_models/ViT-B-32.pt ]; then
    wget -P ./tmp2/pretrained_models https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fashionklip/openai/ViT-B-32.pt
fi
```


## Start training

* Training

```
DATAPATH=./tmp
PRETRAINED_MODEL=./tmp/pretrained_models

MASTER_ADDR=tcp://127.0.0.1:12345

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
        --epochs=20 \
        --workers=0 \
        --model ViT-B/32

```

If you want to only use the FashionKLIP training strategy without loading our provided checkpoint, please remove the argument "resume" and set "openai-pretrained".


* Extracting features

  The resume path can be replaced with your trained model.

  ```
  SAVEFOLDER=./tmp/predictions_finetune

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

  ```

  The output feature files are saved in user-defined $SAVEFOLDER in the following format:
  ("image_id" for images predictions and "query_id" for the texts)

  ```
  {"image_id": "94", "feature": [0.05916735157370567,....]}
  {"query_id": 10, "feature": [0.05950947478413582,....]}

  ```

* Predict and Evaluate

  Note that the prediction and envaluation procedure should be executed after extracting features.
  ```
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

  ```

  The commands will output evaluation files, saved in user-defined "text-out-path" and "image-out-path" in the following format:

  ```
  {"success": true, "score": 12.189873028560887, "scoreJson": {"score": 12.189873028560887, "mean_recall": 12.189873028560887, "r1": 4.60233037169121, "r5": 13.078365665447167, "r10": 18.88892304854429}}
  ```


## Reference
* Fashion-Gen: The Generative Fashion Dataset and Challenge. [[paper](https://arxiv.org/abs/1806.08317)][[website](https://fashion-gen.com/)]
* Learning Transferable Visual Models From Natural Language Supervision. [[paper](https://arxiv.org/abs/1806.08317)][[website](https://fashion-gen.com/)]


## Acknowledgements

The implementation of FashionKLIP relies on resources from OpenAI's [CLIP](https://github.com/openai/CLIP), and the implementation version [OpenCLIP](https://github.com/mlfoundations/open_clip), portions of models/ modelling and tokenizer code are adaptations of official repositories. We thank the original authors for their open-sourcing.


