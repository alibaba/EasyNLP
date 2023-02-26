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
 

# EasyNLP简介
  
随着 BERT、Megatron、GPT-3 等预训练模型在NLP领域取得瞩目的成果，越来越多团队投身到超大规模训练中，这使得训练模型的规模从亿级别发展到了千亿甚至万亿的规模。然而，这类超大规模的模型运用于实际场景中仍然有一些挑战。首先，模型参数量过大使得训练和推理速度过慢且部署成本极高；其次在很多实际场景中数据量不足的问题仍然制约着大模型在小样本场景中的应用，提高预训练模型在小样本场景的泛化性依然存在挑战。为了应对以上问题，PAI 团队推出了 EasyNLP 中文 NLP 算法框架，助力大模型快速且高效的落地。

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


# 主要特性

- **易用且兼容开源**：EasyNLP 支持常用的中文 NLP 数据和模型，方便用户评测中文 NLP 技术。除了提供易用简洁的 PAI 命令形式对前沿NLP算法进行调用以外，EasyNLP 还抽象了一定的自定义模块如 AppZoo 和 ModelZoo，降低NLP 应用的门槛，同时 ModelZoo 里面常见的预训练模型和 PAI 自研的模型，包括知识预训练模型等。EasyNLP 可以无缝接入 huggingface/transformers 的模型，也兼容 EasyTransfer 模型，并且可以借助框架自带的分布式训练框架（基于Torch-Accelerator）提升训练效率。
- **大模型小样本落地技术**：EasyNLP 框架集成了多种经典的小样本学习算法，例如 PET、P-Tuning 等，实现基于大模型的小样本数据调优，从而解决大模型与小训练集不相匹配的问题。此外，PAI 团队结合经典小样本学习算法和对比学习的思路，提出了一种不增添任何新的参数与任何人工设置模版与标签词的方案 Contrastive Prompt Tuning，在 FewCLUE 小样本学习榜单取得第一名，相比 Finetune 有超过 10% 的提升。
- **大模型知识蒸馏技术**：鉴于大模型参数大难以落地的问题，EasyNLP 提供知识蒸馏功能帮助蒸馏大模型从而得到高效的小模型来满足线上部署服务的需求。同时 EasyNLP 提供 MetaKD 算法，支持元知识蒸馏，提升学生模型的效果，在很多领域上甚至可以跟教师模型的效果持平。同时，EasyNLP 支持数据增强，通过预训练模型来增强目标领域的数据，可以有效的提升知识蒸馏的效果。

# 安装

```bash
$ git clone https://github.com/alibaba/EasyNLP.git
$ pip install -r requirements.txt 
$ cd EasyNLP
$ python setup.py install
```

环境要求：Python3.6, PyTorch >= 1.8.

# 快速上手

下面提供一个BERT文本分类的例子，只需要几行代码就可以训练BERT模型：

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

我们也提供了AppZoo的命令行来训练模型，只需要通过简单的参数配置就可以开启训练：
首先需要下载训练集[train.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv)和测试集[dev.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv)，然后开始训练：

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

模型预测命令如下：

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

AppZoo更多示例，详见：[AppZoo文档](https://www.yuque.com/easyx/easynlp/kkhkai).

# ModelZoo
EasyNLP的ModelZoo目前支持如下预训练模型。

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

Please refer to this [readme](https://github.com/alibaba/EasyNLP/blob/master/easynlp/modelzoo/README.md) for the usage of these models in EasyNLP.
Meanwhile, EasyNLP supports to load pretrained models from Huggingface/Transformers, please refer to [this tutorial](https://www.yuque.com/easyx/easynlp/qmq8wh) for details.

# 预训练大模型的落地
EasyNLP提供小样本学习和知识蒸馏，方便用户落地超大预训练模型。

1. [PET](https://github.com/alibaba/EasyNLP/blob/master/examples/fewshot_learning/run_fewshot_pet.sh) (from LMU Munich and Sulzer GmbH): released with the paper [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://aclanthology.org/2021.eacl-main.20.pdf) by Timo Schick and Hinrich Schutze. We have made some slight modifications to make the algorithm suitable for the Chinese language.
2. [P-Tuning](https://github.com/alibaba/EasyNLP/blob/master/examples/fewshot_learning/run_fewshot_ptuning.sh) (from Tsinghua University, Beijing Academy of AI, MIT and Recurrent AI, Ltd.): released with the paper [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf) by Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang and Jie Tang. We have made some slight modifications to make the algorithm suitable for the Chinese language.
3. [CP-Tuning](https://github.com/alibaba/EasyNLP/blob/master/examples/fewshot_learning/run_fewshot_cpt.sh) (from Alibaba PAI): released with the paper [Making Pre-trained Language Models End-to-end Few-shot Learners with Contrastive Prompt Tuning](https://arxiv.org/pdf/2204.00166.pdf) by Ziyun Xu, Chengyu Wang, Minghui Qiu, Fuli Luo, Runxin Xu, Songfang Huang and Jun Huang.
4. [Vanilla KD](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation) (from Alibaba PAI): distilling the logits of large BERT-style models to smaller ones.
5. [Meta KD](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation) (from Alibaba PAI): released with the paper [Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains](https://aclanthology.org/2021.acl-long.236.pdf) by Haojie Pan, Chengyu Wang, Minghui Qiu, Yichang Zhang, Yaliang Li and Jun Huang.
6. [Data Augmentation](https://github.com/alibaba/EasyNLP/tree/master/examples/knowledge_distillation/test_data_aug.sh) (from Alibaba PAI): augmentating the data based on the MLM head of pre-trained language models.


# [CLUE Benchmark](https://www.cluebenchmarks.com/)

EasyNLP提供 [CLUE评测代码](https://github.com/alibaba/EasyNLP/tree/master/benchmarks/clue)，方便用户快速评测[CLUE数据](https://www.cluebenchmarks.com/classification.html)上的模型效果。

```bash
# Format: bash run_clue.sh device_id train/predict dataset
# e.g.: 
bash run_clue.sh 0 train csl
```


根据我们的脚本，可以获得BERT，RoBERTa等模型的评测效果（dev数据）：

(1) bert-base-chinese

| Task | AFQMC  | CMNLI  | CSL    | IFLYTEK | OCNLI  | TNEWS  | WSC    |
|------|--------|--------|--------|---------|--------|--------|--------|
| P    | 72.17% | 75.74% | 80.93% | 60.22%  | 78.31% | 57.52% | 75.33% |
| F1   | 52.96% | 75.74% | 81.71% | 60.22%  | 78.30% | 57.52% | 80.82% |

(2) chinese-roberta-wwm-ext:

| Task | AFQMC  | CMNLI  | CSL    | IFLYTEK | OCNLI  | TNEWS  | WSC    |
|------|--------|--------|--------|---------|--------|--------|--------|
| P    | 73.10% | 80.75% | 80.07% | 60.98%  | 80.75% | 57.93% | 86.84% |
| F1   | 56.04% | 80.75% | 81.50% | 60.98%  | 80.75% | 57.93% | 89.58% |


详细的例子，请参考[CLUE评测示例](https://github.com/alibaba/EasyNLP/tree/master/benchmarks/clue).


# Tutorials

- [自定义文本分类示例](https://www.yuque.com/easyx/easynlp/ds35qn)
- [QuickStart-文本分类](https://www.yuque.com/easyx/easynlp/rxne07)
- [QuickStart-PAI DSW](https://www.yuque.com/easyx/easynlp/gvat1o)
- [QuickStart-MaxCompute/ODPS数据](https://www.yuque.com/easyx/easynlp/vdt5ze)
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

# 修改日志

- EasyNLP v0.0.3 was released in 01/04/2022. Please refer to [tag_v0.0.3](https://github.com/alibaba/EasyNLP/releases/tag/v0.0.3) for more details and history.


# 联系我们

扫描下面二维码加入dingding群，有任何问题欢迎在群里反馈。

<img src="https://cdn.nlark.com/yuque/0/2022/png/2480469/1649324662278-fe178523-6b14-4eff-8f50-7abbf468f751.png?x-oss-process=image%2Fresize%2Cw_357%2Climit_0" width="300"/>

# 参考文献

- DKPLM: https://paperswithcode.com/paper/dkplm-decomposable-knowledge-enhanced-pre
- MetaKD: https://paperswithcode.com/paper/meta-kd-a-meta-knowledge-distillation
- CP-Tuning: https://paperswithcode.com/paper/making-pre-trained-language-models-end-to-end-1
- FashionBERT: https://paperswithcode.com/paper/fashionbert-text-and-image-matching-with

更加详细的解读可以参考我们的 [arxiv 文章](https://paperswithcode.com/paper/easynlp-a-comprehensive-and-easy-to-use)。

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
