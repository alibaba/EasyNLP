# EasyNLP: An Easy-to-use NLP Toolkit

<p align="center">
    <br>
    <img src="https://cdn.nlark.com/yuque/0/2022/png/2480469/1649297935073-2fce0ec9-ec8c-490f-bc25-a8cf50d9918f.png" width="200"/>
    <br>
<p>

<p align="center"> <b> EasyNLP is designed to make it easy to develp NLP applications. </b> </p>
<p align="center">
    <a href="https://www.yuque.com/easyx/easynlp/iobg30">
        <img src="https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258840-bfe6302e-d934-409d-917c-8eab455675c1.svg" height="24">
    </a>
    <a href="https://dsw-dev.data.aliyun.com/#/?fileUrl=https://raw.githubusercontent.com/alibaba/EasyTransfer/master/examples/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb">
        <img src="https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258886-ad896af5-b7da-4ca6-8369-4b14c23cb7a3.svg" height="24">
    </a>
</p>

EasyNLP is an easy-to-use NLP development and application toolkit in PyTorch, first released inside Alibaba in 2021. It is built with scalable distributed training strategies, and supports a comprehensive suite of NLP algorithms for various NLP applications. EasyNLP integrates knowledge distillation and few-shot learning for landing large pre-trained models, and provides a unified framework of model training, inference, and deployment for real-world applications. It has powered more than 10 BUs and more than 20 business scenarios within the Alibaba group. It is seamlessly integrated to Platform of AI (PAI) products, includeing PAI-DSW for development, PAI-DLC for cloud-native training, PAI-EAS for serving, and PAI-Designer for zero-code model training.

# Main Features

- **Easy to use and highly customizable:** In addition to providing easy-to-use and concise commands to call cutting-edge models, it also abstracts certain custom modules such as AppZoo and ModelZoo to make it easy to build NLP applications. It is equiped with the PAI PyTorch distributed training framework TorchAccerator to speed up distributed training.
- **Compatible with open-source libraries:** EasyNLP has APIs to support the training of models from Huggingface/Transformers with the PAI distributed framework. It also supports the pre-trained models in EasyTransfer ModelZoo.
- **Knowledge-injected pre-training:** The PAI team has a lot of research on knowledge-injected pre-training, and builds a knowledge-injected model that wins the first place in the CCF knowledge pre-training competition. EasyNLP integrates these cutting-edge knowledge pre-trained models, including DKPLM and KGBERT.
- **Landing large pre-trained models:** EasyNLP provides few-shot learning capabilities, allowing users to finetune large models with only a few samples to achieve good results. At the same time, it provides knowledge distillation functions to help quickly distill large models to a small and efficient model to faciliate online deployment.
- **Seamless integration to PAI products::** It is seamlessly integrated to [Platform of AI (PAI)](https://www.aliyun.com/product/bigdata/product/learn) products, including PAI-DSW for development, PAI-DLC for cloud-native training, PAI-EAS for serving, and PAI-Designer for zero-coding model training.



# Installation

You can either install from pip 

```bash
$ pip install pai-easynlp (to be released)
```

or setup from the source：

```bash
$ git clone https://github.com/alibaba/EasyNLP.git
$ cd EasyNLP
$ python setup.py install
```
This repo is tested on Python3.6, PyTorch >= 1.8.


# Quick Start
Now let's show how to use just a few lines of code to build a text classification model based on BERT. 

```python
from easynlp.core import Trainer
from easynlp.appzoo import ClassificationDataset, SequenceClassification
from easynlp.utils import initialize_easynlp

args = initialize_easynlp()

train_dataset = ClassificationDataset(
    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    data_file=args.tables,
    max_seq_length=args.sequence_length,
    input_schema=args.input_schema,
    first_sequence=args.first_sequence,
    label_name=args.label_name,
    label_enumerate_values=args.label_enumerate_values,
    is_training=True)

model = SequenceClassification(pretrained_model_name_or_path=args.pretrained_model_name_or_path)
Trainer(model=model,  train_dataset=train_dataset).train()
```

Then you can run the code:
```bash
python main.py \
  --mode train \
  --tables=train_toy.tsv \
  --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
  --first_sequence=sent1 \
  --label_name=label \
  --label_enumerate_values=0,1 \
  --checkpoint_dir=./tmp/ \
  --epoch_num=1  \
  --app_name=text_classify \
  --user_defined_parameters='pretrain_model_name_or_path=bert-tiny-uncased'
```

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
To learn more about the usage of AppZoo, please refer to our [documentation](https://www.yuque.com/easyx/easynlp/psm6fr).



# Tutorials

- [AppZoo-文本向量化](https://www.yuque.com/easyx/easynlp/ts4czl)
- [AppZoo-文本分类/匹配](https://www.yuque.com/easyx/easynlp/vgbopy)
- [AppZoo-序列标注](https://www.yuque.com/easyx/easynlp/vgbopy)
- [AppZoo-GEEP文本分类](https://www.yuque.com/easyx/easynlp/vgbopy)
- [基础预训练实践](https://www.yuque.com/easyx/easynlp/vgbopy)
- [知识预训练实践](https://www.yuque.com/easyx/easynlp/vgbopy)
- [知识蒸馏实践](https://www.yuque.com/easyx/easynlp/vgbopy)
- [小样本学习实践](https://www.yuque.com/easyx/easynlp/vgbopy)
- API docs: [http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/easynlp_docs/html/index.html](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/easynlp_docs/html/index.html)


# Contact Us
Scan the following QR codes to join Dingtalk discussion group. The group discussions are most in Chinese, but English is also welcomed.

<img src="https://cdn.nlark.com/yuque/0/2020/png/2480469/1600310258842-d7121051-32f1-494b-a7a5-a35ede74b6c4.png#align=left&display=inline&height=352&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1178&originWidth=1016&size=312154&status=done&style=none&width=304" width="300"/>

# Reference

- EasyTransfer: https://github.com/alibaba/EasyTransfer
- DKPLM: https://arxiv.org/abs/2112.01047
- MetaKD: https://arxiv.org/abs/2012.01266
- CP-Tuning: https://arxiv.org/abs/2204.00166
- FashionBERT: https://arxiv.org/abs/2005.09801