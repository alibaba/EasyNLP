# X-STA

This project is implemented for the findgs of EMNLP 2023 paper: "Sharing, Teaching and Aligning: Knowledgeable Transfer Learning for Cross-Lingual Machine Reading Comprehension". Our code is based on pytorch and huggingface transformers.

## Requirements
```bash
pip install -r requirement.txt
```

## Quick Start

**NOTE**: Please make sure you have set up the environment correctly. 

1. Download data

To download 3 datasets, please run `bash scripts/download_data.sh` which may take a while.

We use translated training data from XTREME team. Please refere to their [repo](https://github.com/google-research/xtreme) or their [translation](https://console.cloud.google.com/storage/browser/xtreme_translations) directly.

2. Model Training and Evaluation:
```bash
bash scripts/mbert/mlqa.sh
```
