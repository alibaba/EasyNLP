# Official implementation of Few-shot CLIP

## Requirements
### Installation
Our CUDA version is 11.2; Python version is 3.8; our torch versions are as follows:
```bash
torch==1.8.1+cu111
torchaudio==0.8.1
torchvision==0.9.1+cu111
```
You can check your cuda version and install the corresponding torch version follow the [pytorch's official website](https://pytorch.org/get-started/previous-versions/).

Besides, please install dependencies:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Dataset
We have provide the training, validation and test dataset for Visual Entailment, Visual Question Answering and Image Classification(i.e. FGVC, EuroSAT and DTD).

You can download the preprocessed datasets for Visual Entailment by running:
```bash
sh download_ve.sh
```

Download the preprocessed datasets for Visual Question Answering by running:
```bash
sh download_vqa.sh
```

And respectively download the preprocessed dataset for Image Classification(i.e. EuroSAT, DTD, FGVC) by running:
```bash
sh download_eurosat.sh
sh download_dtd.sh
sh download_fgvc.sh
```

It is woth noting: the downloaded data will be placed in the current Path(i.e. `./`) 

## Get Started
### Configs
The running configurations for Visual Entailment, Visual Question Answering, Image Classification(i.e. EuroSAT, FGVC, DTD) 
can be respectively modified in 
`configs/visualentailment.yaml`,`configs/vqa.yaml`,
`configs/eurosat.yaml`, `configs/fgvc.yaml`, `configs/dtd.yaml` including low-resource settings(i.e. few-shot or the number of training samples),
visual encoders, and hyperparamters.

Please set the dataset in the config files befor running. 

In `configs/visualentailment.yaml`, you need to set `root_path`, just as:
```bash
root_path: '/PATH/TO/VisualEntailment/DATASET/snli_ve_%s.tsv'
```

In `configs/vqa.yaml`, you need to set `root_path`, just as:
```bash
root_path: '/PATH/TO/VisualQuestionAnswering/DATASET/%s2014_4_clip.tsv'
```

In `configs/eurosat.yaml`, you need to set `root_path`  just as:
```bash
root_path: '/PATH/STORE/THE/EUROSAT/FOLDERS'
```

In `configs/dtd.yaml`, you need to set `root_path`  just as:
```bash
root_path: '/PATH/STORE/THE/DTD/FOLDERS'
```

In `configs/fgvc.yaml`, you need to set `root_path`  just as:
```bash
root_path: '/PATH/STORE/THE/FGVC/FOLDERS/fgvc-aircraft-2013b'
```

It is worth noting that for the first running, you need to set the `load_cache` and `load_pre_feat` in the configuration files as `True`, and the data caches will be stored in `./caches`. Thereafter you can set both properties as `False` for saving time.   

### Running
For Visual Entailment:
```bash
python main_visualentailment.py --config configs/visualentailment.yaml
```
For Visual Question Ansering:
```bash
python main_vqa.py --config configs/vqa.yaml
```
For DTD:
```bash
python main_dtd_matching.py --config configs/dtd.yaml
```
For EuroSAT:
```bash
python main_eurosat_matching.py --config configs/eurosat.yaml
```
For FGVC
```bash
python main_fgvc_matching.py --config configs/fgvc.yaml
```

## Acknowledgement
This repo benefits from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) and [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter). Thanks for their wonderful works.
