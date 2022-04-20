# TransPrompt

This code is implement for our EMNLP 2021's paper [《TransPrompt：Towards an Automatic Transferable Prompting Framework for Few-shot Text Classification》](https://aclanthology.org/2021.emnlp-main.221/).

Our proposed TransPrompt is motivated by the join of prompt-tuning and cross-task transfer learning. 
The aim is to explore and exploit the transferable knowledge from similar tasks in the few-shot scenario, 
and make the Pre-trained Language Model (PLM) better few-shot transfer learner.
Our proposed framework is accepted by the main conference (long paper track) in EMNLP-2021. 
This code is the default multi-GPU version. We will teach you how to use our code in the following parts.


### 1. Data Preparation

We follow PET to use the same dataset.
Please run the scripts to download the data:
```bash
sh data/download_data.sh
```
or manually download the dataset from [https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). 

Then you will obtain a new director ```data/original```

Our work has two kind of scenario, such as single-task and cross-task.
Different kind scenario has corresponding splited examples. Defaultly, we generate few-shot learning examples, you can also generate full data by edit the parameter (-scene=full).
We only demostrate the few-shot data generation.

#### 1.1 Single-task Few-shot
Please run the scripts to obtain the single-task few-shot examples:
```bash
python3 data_utils/generate_k_shot_data.py --scene few-shot --k 16
```

Then you will obtain a new folder ```data/k-shot-single```

#### 1.2 Cross-task Few-shot
Run the scripts
```bash
python3 data_utils/generate_k_shot_cross_task_data.py --scene few-shot --k 16
```

and you will obtain a new folder ```data/k-shot-cross```

After the generation, the similar tasks will be divided into the same group. We have three groups:
- Group1 (Sentiment Analysis): SST-2, MR, CR
- Group2 (Natural Language Inference): MNLI, SNLI
- Group3 (Paraphrasing): MRPC, QQP

### 2. Have a Training Game

Please follow our papers, we have mask following experiments:
- Single-task few-shot learning: It is the same as LM-BFF and P-tuning, we prompt-tune the PLM only on one task.
- Cross-task few-shot learning: We mix up the similar task in group. At first, we prompt-tune the PLM on cross-task data, then we prompt-tune on each task again.
For the Cross-task Learning, we have two cross-task method:
- (Cross-)Task Adaptation: In one group, we prompt-tune on all the tasks, and then evaluate on each task both in few-shot scenario.
- (Cross-)Task Generalization: In one group, we randomly choose one task for few-shot evaluation (do not used for training), others are used for prompt-tuning.

#### 2.1 Single-task few-shot learning
Take MRPC as an example, please run:
```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/run_single_task.sh
```

#### 2.2 Cross-task few-shot Learning (Task Adaptaion)

Take Group1 as an example, please run the scripts:
```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/run_cross_task_adaptation.sh
```
#### 2.3 Cross-task few-shot Learning (Task Generalization)

Also take Group1 as an example, please run the scripts:
Ps: the unseen task is SST-2.
```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/run_cross_task_generalization.sh
```

### Citation
Our paper citation is:

```
@inproceedings{DBLP:conf/emnlp/0001WQH021,
  author    = {Chengyu Wang and
               Jianing Wang and
               Minghui Qiu and
               Jun Huang and
               Ming Gao},
  title     = {TransPrompt: Towards an Automatic Transferable Prompting Framework
               for Few-shot Text Classification},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican
               Republic, 7-11 November, 2021},
  pages     = {2792--2802},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://aclanthology.org/2021.emnlp-main.221},
  timestamp = {Tue, 09 Nov 2021 13:51:50 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/0001WQH021.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Acknowledgement

The code is developed based on [pet](https://github.com/timoschick/pet).
We appreciate all the authors who made their code public, which greatly facilitates this project. 
This repository would be continuously updated.
