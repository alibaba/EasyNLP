<p align="center">
    <br>
    <img src="https://cdn.nlark.com/yuque/0/2022/png/2480469/1649317417481-d20971cd-cd4f-4e29-8587-c342a128b762.png" width="200"/>
    <br>
<p>
    
<p align="center"> <b> EasyNLP is a Comprehensive and Easy-to-use NLP Toolkit </b> </p>

<div align="center">
    
[![website online](https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258840-bfe6302e-d934-409d-917c-8eab455675c1.svg)](https://www.yuque.com/easyx/easynlp/iobg30)
[![Open in PAI-DSW](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/UI/PAI-DSW.svg)](https://dsw-dev.data.aliyun.com/#/?fileUrl=https://raw.githubusercontent.com/alibaba/EasyTransfer/master/examples/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb)
[![open issues](http://isitmaintained.com/badge/open/alibaba/EasyNLP.svg)](https://github.com/alibaba/EasyNLP/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/alibaba/EasyNLP.svg)](https://GitHub.com/alibaba/EasyNLP/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/alibaba/EasyNLP)](https://GitHub.com/alibaba/EasyNLP/commit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

</div>
 

# EasyNLP [中文介绍](https://github.com/alibaba/EasyNLP/blob/master/README.cn.md)
  
EasyNLP is an easy-to-use NLP development and application toolkit in PyTorch, first released inside Alibaba in 2021. It is built with scalable distributed training strategies and supports a comprehensive suite of NLP algorithms for various NLP applications. EasyNLP integrates knowledge distillation and few-shot learning for landing large pre-trained models, together with various popular multi-modality pre-trained models. It provides a unified framework of model training, inference, and deployment for real-world applications. It has powered more than 10 BUs and more than 20 business scenarios within the Alibaba group. It is seamlessly integrated to [Platform of AI (PAI)](https://www.aliyun.com/product/bigdata/product/learn) products, including PAI-DSW for development, PAI-DLC for cloud-native training, PAI-EAS for serving, and PAI-Designer for zero-code model training.


# Main Features

- **Easy to use and highly customizable:** In addition to providing easy-to-use and concise commands to call cutting-edge models, it also abstracts certain custom modules such as AppZoo and ModelZoo to make it easy to build NLP applications. It is equipped with the PAI PyTorch distributed training framework TorchAccelerator to speed up distributed training.
- **Compatible with open-source libraries:** EasyNLP has APIs to support the training of models from Huggingface/Transformers with the PAI distributed framework. It also supports the pre-trained models in [EasyTransfer](https://github.com/alibaba/EasyTransfer) ModelZoo.
- **Knowledge-injected pre-training:** The PAI team has a lot of research on knowledge-injected pre-training, and builds a knowledge-injected model that wins first place in the CCF knowledge pre-training competition. EasyNLP integrates these cutting-edge knowledge pre-trained models, including DKPLM and KGBERT.
- **Landing large pre-trained models:** EasyNLP provides few-shot learning capabilities, allowing users to finetune large models with only a few samples to achieve good results. At the same time, it provides knowledge distillation functions to help quickly distill large models to a small and efficient model to facilitate online deployment.
- **Multi-modality pre-trained models:** EasyNLP is not about NLP only. It also supports various popular multi-modality pre-trained models to support vision-language tasks that require visual knowledge. For example, it is equipped with CLIP-style models for text-image matching and DALLE-style models for text-to-image generation.

# Technical Articles

We have a series of technical articles on the functionalities of EasyNLP.

- [BeautifulPrompt：PAI推出自研Prompt美化器，赋能AIGC一键出美图](https://zhuanlan.zhihu.com/p/636546340)
- [PAI-Diffusion中文模型全面升级，海量高清艺术大图一键生成](https://zhuanlan.zhihu.com/p/632031092)
- [EasyNLP集成K-Global Pointer算法，支持中文信息抽取](https://zhuanlan.zhihu.com/p/608560954)
- [阿里云PAI-Diffusion功能再升级，全链路支持模型调优，平均推理速度提升75%以上](https://zhuanlan.zhihu.com/p/604483551)
- [PAI-Diffusion模型来了！阿里云机器学习团队带您徜徉中文艺术海洋](https://zhuanlan.zhihu.com/p/590020134)
- [模型精度再被提升，统一跨任务小样本学习算法 UPT 给出解法!](https://zhuanlan.zhihu.com/p/590611518)
- [Span抽取和元学习能碰撞出怎样的新火花，小样本实体识别来告诉你!](https://zhuanlan.zhihu.com/p/590297824)
- [算法 KECP 被顶会 EMNLP 收录，极少训练数据就能实现机器阅读理解](https://zhuanlan.zhihu.com/p/590024650)
- [当大火的文图生成模型遇见知识图谱，AI画像趋近于真实世界](https://zhuanlan.zhihu.com/p/581870071)
- [EasyNLP发布融合语言学和事实知识的中文预训练模型CKBERT](https://zhuanlan.zhihu.com/p/574853281)
- [EasyNLP带你实现中英文机器阅读理解](https://zhuanlan.zhihu.com/p/568890245)
- [跨模态学习能力再升级，EasyNLP电商文图检索效果刷新SOTA](https://zhuanlan.zhihu.com/p/568512230)
- [EasyNLP玩转文本摘要（新闻标题）生成](https://zhuanlan.zhihu.com/p/566607127)
- [中文稀疏GPT大模型落地 — 通往低成本&高性能多任务通用自然语言理解的关键里程碑](https://zhuanlan.zhihu.com/p/561320982)
- [EasyNLP集成K-BERT算法，借助知识图谱实现更优Finetune](https://zhuanlan.zhihu.com/p/553816104)
- [EasyNLP中文文图生成模型带你秒变艺术家](https://zhuanlan.zhihu.com/p/547063102)
- [面向长代码序列的Transformer模型优化方法，提升长代码场景性能](https://zhuanlan.zhihu.com/p/540060701)
- [EasyNLP带你玩转CLIP图文检索](https://zhuanlan.zhihu.com/p/528476134)
- [阿里云机器学习PAI开源中文NLP算法框架EasyNLP，助力NLP大模型落地](https://zhuanlan.zhihu.com/p/505785399)
- [预训练知识度量比赛夺冠！阿里云PAI发布知识预训练工具](https://zhuanlan.zhihu.com/p/449487792)

# Installation

You can setup from the source：

```bash
$ git clone https://github.com/alibaba/EasyNLP.git
$ cd EasyNLP
$ python setup.py install
```

This repo is tested on Python 3.6, PyTorch >= 1.8.

# Quick Start

Now let's show how to use just a few lines of code to build a text classification model based on BERT.

```python
from easynlp.appzoo import ClassificationDataset
from easynlp.appzoo import get_application_model, get_application_evaluator
from easynlp.core import Trainer
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import get_pretrain_model_path

initialize_easynlp()
args = get_args()
user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
pretrained_model_name_or_path = get_pretrain_model_path(user_defined_parameters.get('pretrain_model_name_or_path', None))

train_dataset = ClassificationDataset(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    data_file=args.tables.split(",")[0],
    max_seq_length=args.sequence_length,
    input_schema=args.input_schema,
    first_sequence=args.first_sequence,
    second_sequence=args.second_sequence,
    label_name=args.label_name,
    label_enumerate_values=args.label_enumerate_values,
    user_defined_parameters=user_defined_parameters,
    is_training=True)

valid_dataset = ClassificationDataset(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    data_file=args.tables.split(",")[-1],
    max_seq_length=args.sequence_length,
    input_schema=args.input_schema,
    first_sequence=args.first_sequence,
    second_sequence=args.second_sequence,
    label_name=args.label_name,
    label_enumerate_values=args.label_enumerate_values,
    user_defined_parameters=user_defined_parameters,
    is_training=False)

model = get_application_model(app_name=args.app_name,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    num_labels=len(valid_dataset.label_enumerate_values),
    user_defined_parameters=user_defined_parameters)

trainer = Trainer(model=model, train_dataset=train_dataset,user_defined_parameters=user_defined_parameters,
    evaluator=get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset,user_defined_parameters=user_defined_parameters,
    eval_batch_size=args.micro_batch_size))
    
trainer.train()
```

The complete example can be found [here](https://github.com/alibaba/EasyNLP/blob/master/examples/appzoo_tutorials/sequence_classification/bert_classify/run_train_eval_predict_user_defined_local.sh).

You can also use AppZoo Command Line Tools to quickly train an App model. Take text classification on SST-2 dataset as an example. First you can download the [train.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv), and [dev.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv), then start training:

```bash
$ easynlp \
   --mode=train \
   --worker_gpu=1 \
   --tables=train.tsv,dev.tsv \
   --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
   --first_sequence=sent1 \
   --label_name=label \
   --label_enumerate_values=0,1 \
   --checkpoint_dir=./classification_model \
   --epoch_num=1  \
   --sequence_length=128 \
   --app_name=text_classify \
   --user_defined_parameters='pretrain_model_name_or_path=bert-small-uncased'
```

And then predict:

```bash
$ easynlp \
  --mode=predict \
  --tables=dev.tsv \
  --outputs=dev.pred.tsv \
  --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
  --output_schema=predictions,probabilities,logits,output \
  --append_cols=label \
  --first_sequence=sent1 \
  --checkpoint_path=./classification_model \
  --app_name=text_classify
```

To learn more about the usage of AppZoo, please refer to our [documentation](https://www.yuque.com/easyx/easynlp/kkhkai).

# ModelZoo
EasyNLP currently provides the following models in ModelZoo:

1. PAI-BERT-zh (from Alibaba PAI): pre-trained BERT models with a large Chinese corpus.
2. DKPLM (from Alibaba PAI): released with the paper [DKPLM: Decomposable Knowledge-enhanced Pre-trained Language Model for Natural Language Understanding](https://arxiv.org/pdf/2112.01047.pdf) by Taolin Zhang, Chengyu Wang, Nan Hu, Minghui Qiu, Chengguang Tang, Xiaofeng He and Jun Huang.
3. KGBERT (from Alibaba Damo Academy & PAI): pre-train BERT models with knowledge graph embeddings injected.
4. BERT (from Google): released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423.pdf) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
5. RoBERTa (from Facebook): released with the paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer and Veselin Stoyanov.
6. Chinese RoBERTa (from HFL): the Chinese version of RoBERTa.
7. MacBERT (from HFL): released with the paper [Revisiting Pre-trained Models for Chinese Natural Language Processing](https://aclanthology.org/2020.findings-emnlp.58.pdf) by Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang and Guoping Hu.
8. WOBERT (from ZhuiyiTechnology): the word-based BERT for the Chinese language.
9. FashionBERT (from Alibaba PAI & ICBU): in progress.
10. GEEP (from Alibaba PAI): in progress.
11. Mengzi (from Langboat): released with the paper [Mengzi: Towards Lightweight yet Ingenious
Pre-trained Models for Chinese](https://arxiv.org/pdf/2110.06696.pdf) by Zhuosheng Zhang, Hanqing Zhang, Keming Chen, Yuhang Guo, Jingyun Hua, Yulong Wang and Ming Zhou.
12. Erlangshen (from IDEA): released from the [repo](https://github.com/IDEA-CCNL/Fengshenbang-LM).

Please refer to this [readme](https://github.com/alibaba/EasyNLP/blob/master/easynlp/modelzoo/README.md) for the usage of these models in EasyNLP.
Meanwhile, EasyNLP supports to load pretrained models from Huggingface/Transformers, please refer to [this tutorial](https://www.yuque.com/easyx/easynlp/qmq8wh) for details.

# EasyNLP Goes Multi-modal
EasyNLP also supports various popular multi-modality pre-trained models to support vision-language tasks that require visual knowledge. For example, it is equipped with CLIP-style models for text-image matching and DALLE-style models for text-to-image generation. 


1. [Text-image Matching](https://github.com/alibaba/EasyNLP/blob/master/examples/clip_retrieval/run_clip_local.sh)
2. [Text-to-image Generation](https://github.com/alibaba/EasyNLP/blob/master/examples/text2image_generation/run_appzoo_cli_local.sh)
3. [Image-to-text Generation](https://github.com/alibaba/EasyNLP/blob/master/examples/image2text_generation/run_appzoo_cli_local_clip.sh)


# Landing Large Pre-trained Models
EasyNLP provide few-shot learning and knowledge distillation to help land large pre-trained models.

1. [PET](https://github.com/alibaba/EasyNLP/blob/master/examples/fewshot_learning/run_fewshot_pet.sh) (from LMU Munich and Sulzer GmbH): released with the paper [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://aclanthology.org/2021.eacl-main.20.pdf) by Timo Schick and Hinrich Schutze. We have made some slight modifications to make the algorithm suitable for the Chinese language.
2. [P-Tuning](https://github.com/alibaba/EasyNLP/blob/master/examples/fewshot_learning/run_fewshot_ptuning.sh) (from Tsinghua University, Beijing Academy of AI, MIT and Recurrent AI, Ltd.): released with the paper [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf) by Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang and Jie Tang. We have made some slight modifications to make the algorithm suitable for the Chinese language.
3. [CP-Tuning](https://github.com/alibaba/EasyNLP/blob/master/examples/fewshot_learning/run_fewshot_cpt.sh) (from Alibaba PAI): released with the paper [Making Pre-trained Language Models End-to-end Few-shot Learners with Contrastive Prompt Tuning](https://arxiv.org/pdf/2204.00166.pdf) by Ziyun Xu, Chengyu Wang, Minghui Qiu, Fuli Luo, Runxin Xu, Songfang Huang and Jun Huang.
4. [Vanilla KD](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation) (from Alibaba PAI): distilling the logits of large BERT-style models to smaller ones.
5. [Meta KD](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation) (from Alibaba PAI): released with the paper [Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains](https://aclanthology.org/2021.acl-long.236.pdf) by Haojie Pan, Chengyu Wang, Minghui Qiu, Yichang Zhang, Yaliang Li and Jun Huang.
6. [Data Augmentation](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation/test_data_aug.sh) (from Alibaba PAI): augmentating the data based on the MLM head of pre-trained language models.


# [CLUE Benchmark](https://www.cluebenchmarks.com/)

EasyNLP provides [a simple toolkit](https://github.com/alibaba/EasyNLP/tree/master/benchmarks/clue) to benchmark clue datasets. You can simply use just this command to benchmark CLUE dataset.

```bash
# Format: bash run_clue.sh device_id train/predict dataset
# e.g.: 
bash run_clue.sh 0 train csl
```


We've tested chiese bert and roberta modelson the datasets, the results of dev set are:

(1) bert-base-chinese:

| Task | AFQMC  | CMNLI  | CSL    | IFLYTEK | OCNLI  | TNEWS  | WSC    |
|------|--------|--------|--------|---------|--------|--------|--------|
| P    | 72.17% | 75.74% | 80.93% | 60.22%  | 78.31% | 57.52% | 75.33% |
| F1   | 52.96% | 75.74% | 81.71% | 60.22%  | 78.30% | 57.52% | 80.82% |

(2) chinese-roberta-wwm-ext:

| Task | AFQMC  | CMNLI  | CSL    | IFLYTEK | OCNLI  | TNEWS  | WSC    |
|------|--------|--------|--------|---------|--------|--------|--------|
| P    | 73.10% | 80.75% | 80.07% | 60.98%  | 80.75% | 57.93% | 86.84% |
| F1   | 56.04% | 80.75% | 81.50% | 60.98%  | 80.75% | 57.93% | 89.58% |


Here is the detailed [CLUE benchmark example](https://github.com/alibaba/EasyNLP/tree/master/benchmarks/clue).


# Tutorials

- [自定义文本分类示例](https://www.yuque.com/easyx/easynlp/ds35qn)
- [QuickStart-文本分类](https://www.yuque.com/easyx/easynlp/rxne07)
- [QuickStart-PAI DSW](https://www.yuque.com/easyx/easynlp/gvat1o)
- [QuickStart-MaxCompute/ODPS数据](https://www.yuque.com/easyx/easynlp/vgwe7f)
- [AppZoo-文本向量化](https://www.yuque.com/easyx/easynlp/ts4czl)
- [AppZoo-文本分类/匹配](https://www.yuque.com/easyx/easynlp/vgbopy)
- [AppZoo-序列标注](https://www.yuque.com/easyx/easynlp/qkwqmb)
- [AppZoo-GEEP文本分类](https://www.yuque.com/easyx/easynlp/lepm0q)
- [AppZoo-文本生成](https://www.yuque.com/easyx/easynlp/svde6x)
- [基础预训练实践](https://www.yuque.com/easyx/easynlp/lm1a5t)
- [知识预训练实践](https://www.yuque.com/easyx/easynlp/za7ywp)
- [知识蒸馏实践](https://www.yuque.com/easyx/easynlp/ffu6p9)
- [跨任务知识蒸馏实践](https://www.yuque.com/easyx/easynlp/izbfqt)
- [小样本学习实践](https://www.yuque.com/easyx/easynlp/ochmnf)
- [Rapidformer模型训练加速实践](https://www.yuque.com/easyx/easynlp/bi6nzc)
- API docs: [http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/easynlp_docs/html/index.html](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/easynlp_docs/html/index.html)


# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/alibaba/EasyNLP/blob/master/LICENSE). This toolkit also contains some code modified from other repos under other open-source licenses. See the [NOTICE](https://github.com/alibaba/EasyNLP/blob/master/NOTICE) file for more information.

# ChangeLog

- EasyNLP v0.0.3 was released in 01/04/2022. Please refer to [tag_v0.0.3](https://github.com/alibaba/EasyNLP/releases/tag/v0.0.3) for more details and history.


# Contact Us

Scan the following QR codes to join Dingtalk discussion group. The group discussions are mostly in Chinese, but English is also welcomed.

<img src="https://cdn.nlark.com/yuque/0/2022/png/2480469/1649324662278-fe178523-6b14-4eff-8f50-7abbf468f751.png?x-oss-process=image%2Fresize%2Cw_357%2Climit_0" width="300"/>

# Reference

- DKPLM: https://paperswithcode.com/paper/dkplm-decomposable-knowledge-enhanced-pre
- MetaKD: https://paperswithcode.com/paper/meta-kd-a-meta-knowledge-distillation
- CP-Tuning: https://paperswithcode.com/paper/making-pre-trained-language-models-end-to-end-1
- FashionBERT: https://paperswithcode.com/paper/fashionbert-text-and-image-matching-with

We have [an arxiv paper](https://paperswithcode.com/paper/easynlp-a-comprehensive-and-easy-to-use) for you to cite for the EasyNLP library:

```
@article{easynlp,
  doi = {10.48550/ARXIV.2205.00258},  
  url = {https://arxiv.org/abs/2205.00258},  
  author = {Wang, Chengyu and Qiu, Minghui and Zhang, Taolin and Liu, Tingting and Li, Lei and Wang, Jianing and Wang, Ming and Huang, Jun and Lin, Wei},
  title = {EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing},
  publisher = {arXiv},  
  year = {2022}
}

```
