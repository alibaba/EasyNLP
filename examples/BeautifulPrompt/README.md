# BeautifulPrompt
This project is implemented for the EMNLP Industry Track 2023 paper: "BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis". Our code is based on pytorch and huggingface transformers.

## Data & Models
We released our collected dataset ([train](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/BeautifulPrompt/data.json), [test](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/BeautifulPrompt/test.json)), which includes prompt pairs and various scores, and also released a more extensive [rm_aesthetic dataset](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/BeautifulPrompt/rm_aesthetic.json).

We released the following models:
- [alibaba-pai/pai-bloom-1b1-text2prompt-sd](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd)
- [alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2](https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2)

## Run
### Installation
```bash
conda create -n trlx python=3.10
conda activate trlx

pip install -e .
pip install tensorboardX==2.6.0
```

### Training
```bash
# Step 1
bash scripts/sft.sh
# Step 2
bash scripts/rm_aes.sh
bash scripts/rm_ps.sh
# Step 3
bash scripts/ppo.sh
```

### Evaluation
```bash
pip install pillow
pip install image-reward==1.2
pip install git+https://github.com/openai/CLIP
pip install diffusers

bash scripts/eval.sh
```

## Acknowledgement
This repo benefits from [trlx](https://github.com/CarperAI/trlx). Thanks for their wonderful works.
