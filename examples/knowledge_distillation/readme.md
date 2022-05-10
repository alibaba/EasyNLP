
## 知识蒸馏简介
随着BERT等预训练语言模型在各项任务上都取得了STOA效果，BERT这类模型已经成为 NLP 深度迁移学习管道中的重要组成部分。但 BERT 并不是完美无瑕的，这类模型仍然存在以下两个问题：

1. **模型参数量太大**：BERT-base 模型能够包含一亿个参数，较大的 BERT-large 甚至包含 3.4 亿个参数。显然，很难将这种规模的模型部署到资源有限的环境（例如移动设备或嵌入式系统）当中。
1. **训练/推理速度慢：**在基于 Pod 配置的 4 个 Cloud TPUs（总共 16 个 TPU 芯片）上对 BERT-base 进行训练，或者在 16 个 Cloud TPU（总共 64 个 TPU 芯片）上对 BERT-large 进行训练，每次预训练都需要至少 4 天的时间才能完成。而BERT的推理速度更是严重影响到了需要较高QPS的线上场景，部署成本非常高。

而这个问题，不仅仅是在NLP领域存在，计算机视觉也同样存在，通常来讲有以下三种解决方案：

1. **架构改进**：将原有的架构改进为更小/更快的架构，例如，将 RNN 替换为 Transformer 或 CNN，ALBERT替代BERT等；使用需要较少计算的层等。当然也可以采用其他优化，例如从学习率和策略、预热步数，较大的批处理大小等；
1. **模型压缩**：通常使用量化和修剪来完成，从而能够在架构不变（或者大部分架构不变）的情况下减少计算总量；
1. **知识蒸馏**：训练一个小的模型，使得其在相应任务行为上能够逼近大的模型的效果，如DistillBERT，BERT-PKD，TinyBERT等

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/100470/1595390850746-7c387489-3187-46f8-bf91-f669b93f6be8.png#crop=0&crop=0&crop=1&crop=1&height=250&id=fADJO&margin=%5Bobject%20Object%5D&name=image.png&originHeight=250&originWidth=537&originalType=binary&ratio=1&rotation=0&showTitle=false&size=43896&status=done&style=none&title=&width=537)

### 支持的任务
当前知识蒸馏训练范式仅支持部分下游任务，包括：

1. 文本分类 (Text Classification)
1. 单塔文本匹配 (Single-Tower Text Match)

其他下游任务的支持有待后续更新。

### 主要流程
知识蒸馏的主要流程为：

1. 准备所需的数据集，并提前处理为 EasyTexMiner 支持的 `tsv` 格式（以制表符`\t`分隔的值表）。
1. 选定大规模的预训练模型作为 Teacher Model，并依照其所属的下游任务进行 fine-tuning。详情参考本文档对应章节。
1. 导出训练好的 Teacher Model 的 logits 到文件。
1. 根据需求选定小规模的预训练模型作为 Student Model，并依照知识蒸馏范式进行 fine-tuning。
1. 得到目标模型。

### 参数定义
在常规的下游任务 fine-tuning 命令的基础上，使用知识蒸馏范式需要在 `input_schema` 的末尾追加 logits 条目，并在 `user_defined_parameters` 中显式地启用，以键值对的形式传入所需的参数：

| **参数名** | **类型** | **可选值** | **描述** |
| --- | --- | --- | --- |
| `enable_distillation` | bool | True/False | 是否启用知识蒸馏 |
| `type` | str | vanilla_kd（更多类型有待增加） | 知识蒸馏的类型 |
| `logits_name` | str | 应与 `input_schema` 中一致 | Logits 字段在输入模式中的名称 |
| `logits_saved_path` | str | tsv 文件相对/绝对路径 | Teacher Model 导出的 logits 文件的路径 |
| `temperature` | float | 大于等于 1，一般不超过 10 | 知识蒸馏的温度 |
| `alpha` | float | [0, 1]，一般不大于 0.5 | Teacher Knowledge 在训练过程中的占比 |

具体的 CLI 命令示例如下：
```bash
# SST-2 文本分类 知识蒸馏样例
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.tsv,dev.tsv \
    --input_schema=sent:str:1,label:str:1,logits:float:2 \
    --first_sequence=sent \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./results/small_sst2_student \
    --learning_rate=3e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --save_checkpoint_steps=200 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --user_defined_parameters="
        pretrain_model_name_or_path=${STUDENT_MODEL}
        enable_distillation=True
        type=vanilla_kd
        logits_name=logits
        logits_saved_path=${LOGITS_PATH}
        temperature=5
        alpha=0.2
    "
```
### 
## 基础知识蒸馏示例
本节以英文双句文本分类任务（MRPC）为例，给出完整的知识蒸馏流程命令示例。
可在此下载[训练集](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easytexminer/tutorials/classification/train.tsv)和[验证集](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/easytexminer/tutorials/classification/dev.tsv)。

为了快速测试，样例中使用了尽量精简的超参数设置（#epoch、batch size 等），需要根据实际场景调整。

### 定义所需环境变量
```bash
# GPU device settings
export WORKER_COUNT=1
export WORKER_GPU=1

# Models to be used
export TEACHER_MODEL=bert-large-uncased
export STUDENT_MODEL=bert-small-uncased

# Path to save the fine-tuned models
export TEACHER_CKPT=results/large-sst2-teacher
export STUDENT_CKPT=results/small-sst2-student

# Path to save the teacher logits
export LOGITS_PATH=results/large-sst2-teacher/logits.tsv
```
#### 
### Teacher Fine-tuning
```bash
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.tsv,dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=${TEACHER_CKPT} \
    --learning_rate=3e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --user_defined_parameters="pretrain_model_name_or_path=${TEACHER_MODEL}"
```
#### 
### 导出 Teacher Logits
通过 predict 模式导出 teacher model 对训练集的 logits。
```bash
easynlp \
    --app_name=text_classify \
    --mode=predict \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.tsv \
    --outputs=${LOGITS_PATH} \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --output_schema=logits \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --checkpoint_path=${TEACHER_CKPT} \
    --micro_batch_size=32 \
    --sequence_length=128
```
#### 
### Student 知识蒸馏
注意在 `input_schema` 中追加 logits 字段，类型为 float，数量与任务的标签数保持一致。
```bash
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.tsv,dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1,logits:float:2 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=${STUDENT_CKPT} \
    --learning_rate=3e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --save_checkpoint_steps=200 \
    --sequence_length=128 \
    --micro_batch_size=32 \
    --user_defined_parameters="
        pretrain_model_name_or_path=${STUDENT_MODEL}
        enable_distillation=True
        type=vanilla_kd
        logits_name=logits
        logits_saved_path=${LOGITS_PATH}
        temperature=5
        alpha=0.2
    "
```

### Student 模型预测
```bash
easynlp \
    --app_name=text_classify \
    --mode=predict \
    --worker_gpu=1 \
    --worker_count=1 \
    --tables=dev.tsv \
    --outputs=student_pred.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --output_schema=predictions \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --checkpoint_path=${STUDENT_CKPT} \
    --micro_batch_size=32 \
    --sequence_length=128
```
## 跨任务知识蒸馏MetaKD简介
预训练语言模型的蒸馏往往只关注单一领域的知识，学生模型也只能从对应领域的教师模型中获取知识。知识蒸馏可以让学生模型从多个来自不同领域的教师或跨领域的教师中获取知识，进而帮助目标领域的学生模型训练。但这种方式可能会传递一些来自其他领域的非迁移性知识，这些知识与当前领域无关从而造成模型下降。跨任务知识蒸馏通过元学习的方法获取多个领域的可迁移性知识，提高教师模型在跨领域知识上的泛化性能以提高学生模型的性能。

Meta-KD算法与现有跨任务知识蒸馏不同，借鉴了元学习的思想，首先在多个不同领域数据集上训练一个meta-teacher，获取多个领域的可迁移性知识。在这个meta-teacher的基础上，模型再蒸馏到基于特定任务的学生模型上，取得更好的效果。Meta-KD算法的算法思想如下图所示：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2556002/1647500661600-17f5d7c5-eafc-43e6-b4a5-3c51156b12e9.png#clientId=u0e51cb8b-d6a9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=306&id=u1b7119ae&margin=%5Bobject%20Object%5D&name=image.png&originHeight=782&originWidth=1034&originalType=binary&ratio=1&rotation=0&showTitle=false&size=330592&status=done&style=none&taskId=ud1819c79-eeb8-48cf-a94d-3fb50714266&title=&width=304)
在算法实现中，首先基于不同领域的训练数据，训练meta-teacher。由于不同领域数据的可迁移性不同，我们对每个数据都采用基于Class Centroid的方法计算权重（即为下图的Prototype Score），表示这个数据对于其他各个领域的可迁移性。一般而言，领域特性越小的数据，权重越大。Meta-teacher在领域数据上进行带权重的混合训练。当meta-teacher训练完毕后，我们将这一模型蒸馏到某个特定领域的数据上，充分考虑了多种损失函数的组合。此外，由于meta-teacher不一定在所有领域数据上都具有良好的表现，在蒸馏过程中我们采用了domain-expertise weight衡量meta-teacher对于当前样本预测正确的置信度。Domain-expertise weight较高的样本在蒸馏过程中拥有更高的权重。
![image.png](https://cdn.nlark.com/yuque/0/2022/png/2556002/1647500806787-eb2851dc-8213-40ff-aff7-aa9fc4bd44f1.png#clientId=u0e51cb8b-d6a9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=298&id=uc221ce22&margin=%5Bobject%20Object%5D&name=image.png&originHeight=474&originWidth=1126&originalType=binary&ratio=1&rotation=0&showTitle=false&size=260756&status=done&style=none&taskId=ufc518b45-cdec-4ad2-929e-65961b1fc08&title=&width=309)
Meta-KD算法的细节可以参考论文_Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains _（ACL-IJCNLP 2021）[[链接]](https://aclanthology.org/2021.acl-long.236.pdf)。
### 环境准备
完整代码位于`EasyNLP/examples/knowledge_distillation/metakd`
下载示例数据集并划分：
```bash
cd data
if [ ! -f ./SENTI/dev.tsv ];then
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/domain_sentiment_data.tar.gz
tar -zxvf domain_sentiment_data.tar.gz
fi
cd ..

if [ ! -f data/SENTI/dev.tsv ];then
python generate_senti_data.py
fi


```
### 预处理示例数据集
产生训练所需meta-weight并统一测试集格式：
```bash
if [ ! -f data/SENTI/train.embeddings.tsv ];then
python extract_embeddings.py \
--bert_path ~/.easynlp/modelzoo/bert-base-uncased \
--input data/SENTI/train.tsv \
--output data/SENTI/train.embeddings.tsv \
--task_name senti --gpu 7
fi

if [ ! -f data/SENTI/train_with_weights.tsv ];then
python generate_meta_weights.py \
data/SENTI/train.embeddings.tsv \
data/SENTI/train_with_weights.tsv \
books,dvd,electronics,kitchen
fi

if [ ! -f data/SENTI/dev_standard.tsv ];then
python generate_dev_file.py \
--input data/SENTI/dev.tsv \
--output data/SENTI/dev_standard.tsv
fi
```
### 训练meta-teacher
训练时需要指定use_sample_weight和use_domain_loss为Ture并设定domain_loss_weight的值。
```bash
model=bert-base-uncased
DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
python -m torch.distributed.launch $DISTRIBUTED_ARGS meta_teacher_train.py \
--mode train \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--second_sequence=text_b \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./tmp/meta_teacher/ \
--learning_rate=3e-5  \
--epoch_num=1  \
--random_seed=42 \
--logging_steps=20 \
--save_checkpoint_steps=50 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="
  pretrain_model_name_or_path=$model
  use_sample_weights=True
  use_domain_loss=True
  domain_loss_weight=0.5
                          "
```
### 蒸馏对应领域的学生模型
蒸馏对应两个阶段，第一阶段为拟合教师模型的中间层输出，第二阶段通过蒸馏损失函数训练学生模型。
第一阶段需要指定教师模型的保存路径`teacher_model_path`， 将`distill_stage`设置为`first`。此外，第一阶段蒸馏的`checkpoint_dir`将作为第二阶段蒸馏的模型输入`pretrain_model_name_or_path`
第二阶段同样需要制定教师模型的保存路径，将将`distill_stage`设置为`second`。同时确保`pretrain_model_name_or_path`为一阶段的模型保存位置。
```bash
model=bert-tiny-uncased

# In domain_sentiment_data, genre is one of ["books", "dvd", "electronics", "kitchen"]
genre=books
cd ${cur_path}

# 1. Distillation pretrain
DISTRIBUTED_ARGS="--nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
# Pretrained distillation
python -m torch.distributed.launch $DISTRIBUTED_ARGS meta_student_distill.py \
--mode train \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--second_sequence=text_b \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./tmp/$genre/meta_student_pretrain/ \
--learning_rate=3e-5  \
--epoch_num=10  \
--random_seed=42 \
--logging_steps=20 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="
      pretrain_model_name_or_path=$model
      teacher_model_path=./tmp/meta_teacher/
      domain_loss_weight=0.5
      distill_stage=first
      genre=$genre
      T=2
      "

# 2. Finetune
pretrained_path="./tmp/$genre/meta_student_pretrain/"
python -m torch.distributed.launch $DISTRIBUTED_ARGS meta_student_distill.py \
--mode train \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--second_sequence=text_b \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./tmp/$genre/meta_student_fintune/ \
--learning_rate=3e-5  \
--epoch_num=10  \
--random_seed=42 \
--logging_steps=20 \
--save_checkpoint_steps=50 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="
        pretrain_model_name_or_path=$pretrained_path
        teacher_model_path=./tmp/meta_teacher/
        domain_loss_weight=0.5
        distill_stage=second
        genre=$genre
        T=2
        "

# 3. Evalute
Student_model_path=./tmp/$genre/meta_student_fintune/
python main_evaluate.py \
--mode evaluate \
--tables=data/SENTI/train_with_weights.tsv,data/SENTI/dev_standard.tsv \
--input_schema=guid:str:1,text_a:str:1,text_b:str:1,label:str:1,domain:str:1,weight:str:1 \
--first_sequence=text_a \
--label_name=label \
--label_enumerate_values=positive,negative \
--checkpoint_dir=./tmp/meta_teacher/ \
--learning_rate=3e-5  \
--epoch_num=1  \
--random_seed=42 \
--logging_steps=20 \
--sequence_length=128 \
--micro_batch_size=16 \
--app_name=text_classify \
--user_defined_parameters="pretrain_model_name_or_path=$Student_model_path
                              genre=$genre"
```
预测时请确保测试集的格式与训练集文件`train_with_weights.tsv`一致。
