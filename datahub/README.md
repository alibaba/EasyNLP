EasyNLP提供常见的**中文**数据集的下载，同时提供如下**接口加载**和处理中文数据。

# 目录
[使用DataHub数据进行训练](#使用DataHub数据进行训练)  
[预训练数据](#预训练数据)  
[通用NLU数据](#通用NLU数据)  
[中文文本匹配/问答数据](#中文文本匹配/问答数据)  
[中文文本分类](#中文文本分类)  
[中文序列标注](#中文序列标注)  
[文本生成数据（摘要、对话等）](#文本生成数据（摘要/对话等）)  
[知识图谱](#知识图谱)  
[知识任务数据](#知识任务数据)  
[零样本学习](#零样本学习)  
[小样本学习](#小样本学习)  
[多模态-检索数据](#多模态-检索数据)  
[多模态-生成数据](#多模态-生成数据)  


# 使用DataHub数据进行训练
对于已经过huggingface或easynlp预处理的数据，您可以直接使用以下方式加载与训练:
```python
from easynlp.core import Trainer
from easynlp.appzoo import GeneralDataset, SequenceClassification, load_dataset
from easynlp.utils import initialize_easynlp
args = initialize_easynlp()
row_data = load_dataset('clue', 'afqmc')["train"]
train_dataset = GeneralDataset(dataset, args.pretrained_model_name_or_path, args.sequence_length)
model = SequenceClassification(pretrained_model_name_or_path=args.pretrained_model_name_or_path, num_labels=train_dataset.num_label)
Trainer(model=model, train_dataset=train_dataset).train()
```

使用DataHub的数据可以大幅度减少输入参数，保存上述代码并使用以下脚本开始训练程序:
```bash
python main.py \
 --mode train \
 --worker_gpu=1 \
 --checkpoint_dir=./tmp/ \
 --epoch_num=1 \
 --user_defined_parameters='pretrain_model_name_or_path=hfl/chinese-roberta-wwm-ext'
```


对于新的数据集，您可以使用下面方式加载（以文本分类为例）：
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

model = SequenceClassification(pretrained_model_name_or_path=args.pretrained_model_name_or_path, num_labels=train_dataset.num_label)
Trainer(model=model, train_dataset=train_dataset).train()
```

具体的例子详见[quick start](https://github.com/alibaba/EasyNLP/blob/master/examples/quick_start_user_defined/main.py)。


# 预训练数据
| **数据** | **描述** | **数据格式** |
| --- | --- | --- |
| Wudao（中文）（[链接](https://resource.wudaoai.cn/home?ind&name=WuDaoCorpora%202.0&id=1394901288847716352)） | 5900万文本数据 | json格式，包括Topic（标题），Text（正文） |
| WuDaoMM-base（[链接](https://resource.wudaoai.cn/home )） | WuDao大数据的一个子数据集，共500万图文对。支持了文澜、Cogview 等中文多模态预训练） | json格式，数据包含19个大类，分别为：能源、表情、工业、医疗、风景、动物、新闻、花卉、教育、艺术、人物、科学、大海、树木、汽车、社交、科技、运动等，单类别数据约7万~40万左右。每个json文件包括name，tag，图片url，和captions。 |
| CLUE-news2016（[直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/pretrain_corpus/news2016zh.zip)） | CLUE社区收集的250万篇新闻，含关键词和描述 | 8G新闻语料，分成两个上下两部分，总共有2000个小文件 |
| CLUE-webText2019（[直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/pretrain_corpus/webtext2019zh.zip)） | CLUE社区收集的419万个高质量社区问答，适合训练通用预训练模型或者问答模型 | 社区互动3G语料，包含3G文本，总共有900多个小文件 |
| CLUE-wiki2019（[直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/pretrain_corpus/wiki_zh_2019.zip)） | CLUE社区的维基百科语料，104万个结构良好的中文词条 | 1.1G左右文本，包含300左右小文件 |
| CLUE-baike2018qa（[直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/pretrain_corpus/baike2018qa.zip)） | CLUE社区收集的百科问答数据，150万个问答数据，包括问答数据和问题类型。数据集划分：数据去重并分成三个部分。训练集：142.5万；验证集：4.5万，测试集无。 | 含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别，其中频率达到或超过10次的类别有434个。 |
| CLUE-translation2019（[直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/pretrain_corpus/translation2019zh.zip)） | CLUE社区收集的中文机器翻译数据，520万个中英文句子对。数据集划分：数据去重并分成三个部分。训练集：516万；验证集：3.9万 | 每一个对，包含一个英文和对应的中文。中文或英文，多数情况是一句带标点符号的完整的话。对于一个平行的中英文对，中文平均有36个字，英文平均有19个单词 |
| 互联网图片库2.0（SogouP2.0）（[链接](http://www.sogou.com/labs/resource/p2.php)） | 来自搜狗识图搜[http://pic.sogou.com/shitu/index.html](http://pic.sogou.com/shitu/index.html) 索引的部分数据。其中收集了包括人物、动物、建筑、机械、风景、运动等类别，总数高达1000万张图片。图片库还包括了一个识图搜索结果人工标注集合，用于训练和评测。 | 共包括三个文件：Meta_Data,Original_Pic,Evaluation_Data。其中Meta_Data存储图片的相关元数据；Original_Pic中存储图片的原图；Evaluation_Data是识图搜索结果的人工标注集合。对于每张图片，搜狗给出了图片的原图文件、图片的URL、图片所在网页的URL、图片所在网页中的Surrounding Text文本、同主题系列图片等信息。 |

# 通用NLU数据
| **数据** | **id** | **描述** | **数据格式** |
| --- | --- | --- | --- |
| AFQMC ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/afqmc.zip)) | clue/afqmc | 蚂蚁金融语义相似度 数据量：训练集（34334）验证集（4316）测试集（3861) | 任务：文本分类， json格式，包括句子1，句子2，和标签，样例：{"sentence1": "xxx", "sentence2": "xxx", "label": "0"} |
| TNEWS1.1 ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/tnews.zip)) | clue/tnews | 今日头条中文新闻（短文本）分类 数据量：训练集(53,360)，验证集(10,000)，测试集(10,000) | 任务：文本分类，json格式，包括id，sentence，和label |
| IFLYTEK ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/iflytek.zip)) | clue/iflytek | 长文本分类 数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)  | json格式，包括分类ID，分类名称，和新闻文本，样例：{"label": "102", "label_des": "news_entertainment", "sentence": "xxx"} |
| WSC1.1 ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/cluewsc.zip)) | clue/cluewsc2020 | 代词消歧 （小样本文本分类）数据量：训练集(1000)，验证集(300)，测试集(300)  | json格式，包括span2_index, span1_index, span2_text, span1_text, id, text （原始文本），span2为原始文本中的指代词，span1为指代的内容 |
| CSL ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/csl.zip)) | clue/csl | 论文关键词识别（文本分类）数据量：训练集(20,000)，验证集(3,000)，测试集(3,000)  | json格式，包括id，abst，label, 和keyword，其中label取值为0/1 |
| CMNLI ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/cmnli.zip)) | clue/cmnli | 语言推断任务 CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等。该数据集可用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。每一条数据有三个属性 | json格式，包括sentence1，sentence2，和label，其中label标签有三种：neutral，entailment，contradiction |

# 中文文本匹配/问答数据
| **数据** | **id** | **描述** | **数据格式** |
| --- | --- | --- | --- |
| OCNLI_50k（[下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/ocnli_50k.zip)）| clue/ocnli | 中文自然语言推理 50k | json域为：level，sentence1, sentence2, label, label0, label1, label2, label3, label4, genre, prem_id, id |
| OCNLI_30k（[下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/ocnli_30k.zip)）| clue/ocnli | 中文自然语言推理 30k | json域为：level，sentence1, sentence2, label, label0, label1, label2, label3, label4, genre, prem_id, id |
| QBQTC（[下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/qbqtc.zip)） | qbqtc<br>(easynlp) | QQ浏览器搜索匹配数据 200k data | json域为: id, query, title, label |
| CMNLI（[下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/cmnli.zip)） | clue/cmnli | XNLI和MNLI (多领域数据）400k data | json域为: sentence1, sentence2, label |
| cMedQA2（[下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/cmedqa2.zip)） | none | 医疗问答数据 10.8k | 分为正文内容和索引，正文(问题)(问题)格式为csv(question_id, conten回答(ans_id, question_id,content)索引为csv(question_id, ans_id, cnt, lable)t), 回答(ans_id, question_id,content)索引为csv(question_id, ans_id, cnt, lable) |
| CAIL2019相似案例匹配大赛([下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/CAIL2019-SCM.zip)) | cail2018<br>(2018version) | 文书事实描述匹配数据集 | json域为: A， B，C, label |
| ChineseTextualInference([下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/chinesetextualinference.zip)) | none | 中文文本推断项目,包括88万文本蕴含中文文本蕴含数据集的翻译与构建 | tsv格式，三个域为sentence1，sentence2，label |
| ChineseSTS([下载链接](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/chinesests.zip)) | none | STS 中文文本语义相似度语料库建设，相似度为0-5，数值越高相似度越高 | tsv格式，5个域为：index1，sentence1， index2， sentence2， 相似度 |

# 中文文本分类
| **数据** | **id** | **描述** | **数据格式** |
| --- | --- | --- | --- |
| TNEWS1.1 | clue/tnews | 详见通用NLU任务 |  |
| IFLYTEK | clue/iflytek | 详见通用NLU任务 |  |
| AFQMC | clue/afqmc | 详见通用NLU任务 |  |
| WSC1.1 | clue/wsc | 详见通用NLU任务 |  |
| CSL | clue/csl | 详见通用NLU任务 |  |
| tc-corpus-answer ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_classification/tc-corpus-answer.rar)) | none | 复旦大学计算机信息与技术系国际数据库中心自然语言处理小组，训练9804篇（train），测试9833篇（answer），标签为20个类别 | 压缩包，包括train.rar, answer.rar |
| Sogou-CA([链接](http://www.sogou.com/labs/resource/ca.php)) | sogou_news<br>(ca+cs) | 数据来自若干新闻站点2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据 | 压缩包 |
| Sogou-CS([链接](http://www.sogou.com/labs/resource/cs.php)) | sogou_news<br>(ca+cs) | 数据来源为搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据 | 压缩包 |
| online_shopping | none | 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店。来源：SophonPlus | rar格式，10 个类别（书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店），共 6 万多条评论数据，正、负向评论各约 3 万条，包括label和review两个字段 |
| weibo_senti | none | 10 万多条，带情感标注 新浪微博 | csv格式，正负向评论约各 5 万条 |
| simplifyweibo | none | 36 万多条，带情感标注 新浪微博 | csv格式，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条 |
| dmsc_v2 | none | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 | csv格式，包含movieid，title，和tile cn，即id和中英文标题 |
| yf_dianping | none | 24 万家餐馆，54 万用户，440 万条评论/评分数据 | csv格式，包括userid，restid（餐馆id），rating（评分），rating_env（环境评分）,rating_flavor（口味评分）, rating_service（服务评分）, timestamp, comment |
| yf_amazon | none | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 | csv格式，包括userid，product id, rating, timestamp, title, comment |
| ChnSentiCorp | seamew/ChnSentiCorp | 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论，来源：SophonPlus | csv格式，包括label和review两个字段，label包括正向和负向。数据来源：携程网, 原数据集由谭松波 老师整理的一份数据集 |
| waimai | XiangPan/waimai_10k | 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条。来源：SophonPlus | csv格式，包括label和review两个字段，label包括正向和负向 |

# 中文序列标注
| **数据** | **id** | **描述** | **数据格式** |
| --- | --- | --- | --- |
| Chinese Treebank**(**[**链接**](https://verbs.colorado.edu/chinese/ctb.html)**)** |  | 词性标注任务 | 每个单词对应的词性信息 |
| ResumeNER**(**[**链接**](https://github.com/jiesutd/LatticeLSTM/tree/master/ResumeNER)**)** | msra_ner |中文命名实体标注任务，微博数据构造（ACL2018） | 每个字对应一行，同时包含标签，句子以空行间隔 |
| People's Daily**(**[**链接**](https://github.com/zjy-ucas/ChineseNER/tree/master/data)**)** | chinese_ner<br>(easynlp) |中文命名实体标注任务，人民日报 | 每个字对应一行，同时包含标签，句子以空行间隔 |
| CNMER**(**[**链接**](https://github.com/yhzbit/CNMER/tree/master/data)**)** | cnmer<br>(easynlp) | 中文医学实体识别数据集，实体包括身体部位、症状体征、检查、疾病以及治疗。 | 每个字对应一行，同时包含标签，句子以空行间隔 |
| CCKS2018数据**(**[**链接**](https://github.com/MenglinLu/Chinese-clinical-NER/tree/master/data)**)** | | 识别疾病和诊断、解剖部位、影像检查、实验室检验、手术和药物6种命名实体 | 例句与标注文件分开 |
| CCKS2019数据**(**[**链接**](http://openkg.cn/dataset/yidu-s4k)**)** | | 识别中文医学命名实体 | json |
| SRL**(**[**链接**](https://catalog.ldc.upenn.edu/LDC2013T19)**)** | | 中文语义角色标注任务(OntoNotes Release 5.0一部分) | 需要进一步处理 |
| OntoNotes**(**[**链接**](https://catalog.ldc.upenn.edu/LDC2013T19)**)** | | 中文命名实体识别任务 总共 15740 | 有18种命名实体类型；每条sample包含3条数据项：输入文本和标注出来的实体位置和对应的实体类型。 |
| MSRA **(**[**链接**](https://catalog.ldc.upenn.edu/LDC2013T19)**)** | msra_ner | 中文命名实体识别任务训练集：46675 | 有3种命名实体类型；每条sample包含3条数据项：输入文本和标注出来的实体位置和对应的实体类型。 |

# 文本生成数据（摘要/对话等）
| **数据** | **描述** | **数据格式** |
| --- | --- | --- |
| Dureader[(链接)](https://arxiv.org/pdf/1711.05073.pdf) [下载](https://dataset-bj.cdn.bcebos.com/dureader/dureader_preprocessed.zip)| 百度中文阅读理解数据集（改造成：问题生成任务）200,000 问题/1,000,000 文档 | 每条sample包含5个数据项：question：输入的问题；question type：问题类型（yes-no，entity-fact等）；answer：问题的对应答案；support sentence：答案在文档中的支持句；document：输入文档 |
| DureaderQG[(链接)](https://www.luge.ai/#/luge/dataDetail?id=8) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_generation/question_generation/DuReaderQG.zip)| 从DuReader数据集中抽取的部分数据作为问题生成数据集，训练集14500，开发集1000 | 每条sample包含4个数据项：context：上下文信息；question：问题；answer：问题的对应答案；id：编号 |
| KdConv[(链接)](https://github.com/thu-coai/KdConv) | 多领域对话生成任务 总共：4,500对话轮次 | 每条sample包含2条数据项：user1-userN用户的对话记录；knowledge triple：用户对话记录文本中对应识别出来的知识三元组 |
| WMT20-enzh[(链接)](https://aclanthology.org/2020.wmt-1.30.pdf) | 中英文机器翻译任务| 每条sample包含2条数据项：源语言和目标语言对应的翻译文本。 |
| MTG-question-generation[(链接)](https://mtg-benchmark.netlify.app) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_generation/question_generation/MTG-question.zip) | 多语言文本生成数据集，包含英语、德语、法语、西班牙语、中文；由于是通过翻译得到的样本，数据质量存在一定问题 | 数据由各个语言的source（src）文件和target（trg）文件组成，每行为一条样本 |
| MTG-story-generation[(链接)](https://mtg-benchmark.netlify.app) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_generation/story_generation/MTG-story.zip) | 多语言文本生成数据集，包含英语、德语、法语、西班牙语、中文；由于是通过翻译得到的样本，数据质量存在一定问题| 数据由各个语言的source（src）文件和target（trg）文件组成，每行为一条样本 |
| MTG-title-generation[(链接)](https://mtg-benchmark.netlify.app) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_generation/title_generation/MTG-title.zip) | 多语言文本生成数据集，包含英语、德语、法语、西班牙语、中文；由于是通过翻译得到的样本，数据质量存在一定问题| 数据由各个语言的source（src）文件和target（trg）文件组成，每行为一条样本 |
| MTG-summarization[(链接)](https://mtg-benchmark.netlify.app) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/MTG-summarization.zip) | 多语言文本生成数据集，包含英语、德语、法语、西班牙语、中文；由于是通过翻译得到的样本，数据质量存在一定问题| 数据由各个语言的source（src）文件和target（trg）文件组成，每行为一条样本 |
| AdvertiseGen[(链接)](https://www.luge.ai/#/luge/dataDetail?id=9) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_generation/advertisement_generation/AdvertiseGen.zip) | AdvertiseGen以商品网页的标签与文案的信息对应关系为基础构造，是典型的开放式生成任务，在模型基于key-value输入生成开放式文案时，与输入信息的事实一致性需要得到重点关注。| 任务描述：给定商品信息的关键词和属性列表kv-list，生成适合该商品的广告文案adv；数据规模：训练集114k，验证集1k；数据来源：清华大学CoAI小组； |
| chat[(链接)](https://github.com/codemayq/chinese_chatbot_corpus) [直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_generation/chitchat/raw_chat_corpus-1.zip) | 多源对话数据集 | chatterbot；豆瓣多轮；PTT八卦语料；青云语料；电视剧对白语料；贴吧论坛回帖语料；微博语料；小黄鸡语料 |
| education([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/education_data.zip)) | **标题生成任务（短文本生成式摘要）** 教育培训行业摘要数据是github作者wonderfulsuccess整理，数据主要由教育培训行业主流垂直媒体的历史文章 总数量：24423个样本；摘要：平均字数 52 正文：平均字数 2016 | json格式包括title，content.其中content为新闻正文 title为新闻的标题 |
| lcsts([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/lcsts_data.zip)) | **标题生成任务（短文本生成式摘要)** lcsts摘要数据是哈尔滨工业大学整理，基于新闻媒体在微博上发布的新闻摘要创建了该数据集 总数量：2108915个样本；摘要：平均字数 18 正文：平均字数 104 | json格式包括title，content.其中content新闻正文title为新闻的标题 |
| thucnews([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/thucnews_data.zip)) | **标题生成任务（短文本生成式摘要)** 清华新闻（THUCNews）数据是清华大学自然语言处理实验室整理，根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成.利用其正文与标题，可以构成新闻标题生成数据 总数量：830749个样本；标题：平均字数 19 正文：平均字数 892 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| SogouCS([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/sohu_data.zip)) | **标题生成任务（短文本生成式摘要)** 搜狗新闻（SogouCS）数据是搜狗实验室整理，来自搜狐新闻2012年6月—7月 利用其正文与标题，可以构成新闻标题生成数据。整理后数据信息如下：总数量：1245835个样本；标题：平均字数 17 正文：平均字数 494 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| nlpcc2017([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/nlpcc_data.zip)) | **标题生成任务（短文本生成式摘要)** nlpcc2017摘要数据是2017年NLPCC比赛Task3任务的数据集。总数量：50000个样本；摘要：平均字数 44 正文：平均字数 990 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| shence([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/shence_data.zip)) | **标题生成任务（短文本生成式摘要)** 神策杯2018摘要数据是“神策杯”2018高校算法大师赛的比赛数据 总数量：108089个样本；摘要：平均字数 24 正文：平均字数 1055 | json格式包括title，content.其中content新闻正文title为新闻的标题 |
| weixin([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/weixin_data.zip)) | **标题生成任务（短文本生成式摘要)** 微信公众号摘要数据是github作者nonamestreet整理 整理后数据信息如下： 总数量：712826个样本；标题：平均字数 22 正文：平均字数 1499 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| new2016zh([直接下载part1](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/new2016zh_data_1.json) [直接下载part2](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/new2016zh_data_2.json) [直接下载part3](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/new2016zh_data_3.json) [直接下载part4](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/new2016zh_data_4.json) [直接下载part5](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/new2016zh_data_5.json)) | **标题生成任务（短文本生成式摘要)** news2016zh新闻数据是 CLUEbenchmark整理 总数量：2317427个样本；标题：平均字数 20 正文：平均字数 1250 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| weibo（[直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/weibo_data.zip)） | 数据来源于新浪微博，由He Zhengfang整理，整理后数据信息如下：总数量：450295个样本；标题：平均字数 18，字数标准差 5，最大字数 95，最小数字 4；正文：平均字数 123，字数标准差 30，最大字数 1873，最小数字 100； | json格式文件，包含：新闻标题（title），新闻正文（article），和新闻摘要（summary） |
| CNewSum（[链接](https://dqwang122.github.io/projects/CNewSum/)）[直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/CNewSum.zip) | 中文大规模中长文档摘要数据，包含304307篇文档以及对应的人工书写的摘要。文档平均长度为730，摘要平均长度为35。 | jsonl格式文件，包含：样本数据以及完整的训练和测试数据 |
| clts（[链接](https://link.springer.com/chapter/10.1007/978-3-030-60450-9_42)）[直接下载](https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/text_summarization/clts.zip) | 中文大规模长文档摘要数据，包含148300篇文档以及对应的摘要。文档平均长度为1363，摘要平均长度为58。 | json格式文件，包含：新闻标题（title），新闻正文（article），和新闻摘要（summary） |


# 知识图谱
| **数据** | **描述** | **数据格式** | 
| --- | --- | --- |
| CN-DBpedia([2015版dump数据和mention2entity](http://openkg.cn/dataset/cndbpedia))([开放API](http://kw.fudan.edu.cn/apis/cndbpedia/)) | 中文通用知识图谱 来源：中文百科（如百度百科、互动百科、中文维基百科等）包含900万+的百科实体以及6700万+的三元组关系。 | txt格式，每行一条数据，每条数据是一个(实体名称，属性名称，属性值)的三元组，中间用tab分隔: 实体名称 \\t 属性名称 \\t 属性值 |
| AliOpenKG([下载链接-需申请](https://kg.alibaba.com/index.html)) | 开放数字商业知识图谱 包含18亿的三元组，多达67万的核心概念，2681类关系。 | subject \\t predicate \\t objec 例如：link1  \\t link2 \\t 正装长袖衬衫 |
| Zhishi.me([dump-turtle格式和jsonld格式](http://openkg.cn/dataset/zhishi-me-dump)) | 中文通用知识图谱 来源：中文百科（如百度百科、互动百科、中文维基百科等）| json或者turtle格式|
| XLore([开放API](https://xloreapi.docs.apiary.io/#reference/0/0)) | 多语言通用知识图谱 来源：中英文维基和百度百科 包含2615万实例，235万概念，51万属性。 | api接口 | 

# 知识任务数据
| **数据** | **描述** | **数据格式** | 
| --- | --- | --- | 
| **FinRE** **(**[**链接**](https://github.com/thunlp/Chinese_NRE/tree/master/data/FinRE)**)** | 金融领域新闻关系抽取 18000+样本 | 44种关系分类类型，每条sample包含4个数据项：输入文本；待分类的头、尾实体位置；关系类型 |
| SanWen **(**[**链接**](https://github.com/thunlp/Chinese_NRE/tree/master/data/FinRE)**)** | 中文文献关系抽取 | 9种关系分类类型，每条sample包含4个数据项：输入文本；待分类的头、尾实体位置；关系类型 |
| OntoNotes **(**[**链接**](https://catalog.ldc.upenn.edu/LDC2013T19)**)** | 中文命名实体识别任务 总共 15740 | 有18种命名实体类型；每条sample包含3条数据项：输入文本和标注出来的实体位置和对应的实体类型。 |
| MSRA **(**[**链接**](https://catalog.ldc.upenn.edu/LDC2013T19)**)** | 中文命名实体识别任务 训练集：46675 | 有3种命名实体类型；每条sample包含3条数据项：输入文本和标注出来的实体位置和对应的实体类型。 |

# 零样本学习
| 数据 | 描述 | 数据格式 |
| --- | --- | --- |
| EPRSTMT([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/eprstmt.zip)) |  电商产品评论情感分析数据集 数据量：训练集（32），验证集（32），公开测试集（610），测试集（753），无标签语料（19565） | json格式，包括id，句子，和标签，样例：{"id": "xxx", "sentence": "xxx", "label": "xxx"} |
| CSLDCP([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/csldcp.zip)) |  中文科学文献学科分类数据集 数据量：训练集（536），验证集（536），公开测试集（1784），测试集（2999），无标签语料（67） | json格式，包括id，sentence，和label |
| TNEWS([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/tnews.zip)) |  今日头条中文新闻（短文本）分类数据集 该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游、教育、金融、军事等。  | json格式，包括分分类ID，分类名称，新闻字符串（仅含标题）。 |
| IFLYTEK([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/iflytek.zip)) |  长文本分类数据集 该数据集关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。  | json格式，每一条数据有三个属性，从前往后分别是 类别ID，类别名称，文本内容。 |
| OCNLI([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/ocnli.zip)) | 中文原版自然语言推理数据 数据量：训练集（32），验证集（32），公开测试集（2520），测试集（3000），无标签语料（20000） | json格式，包括level，sentence1，sentence2, label，label0，label1，label2，label3，label4，genre，prem_id和id。 |
| BUSTM([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/bustm.zip)) |  小布助手对话短文本匹配数据集 数据量：训练集（32），验证集（32），公开测试集（1772），测试集（2000），无标签语料（4251）  | json格式，包括id，sentence1，sentence2，和label |
| ChID ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/chid.zip)) |  成语阅读理解填空 数据量：训练集（42），验证集（42），公开测试集（2002），测试集（2000），无标签语料（7585）  | json格式，包括id，candidates，content，和answer |
| CSL ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/csl.zip)) |  论文关键词识别 数据量：训练集（32），验证集（32），公开测试集（2828），测试集（3000），无标签语料（19841）  | 每一条数据有四个属性，从前往后分别是 数据ID，论文摘要，关键词，真假标签。  |
| CLUEWSC ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_zeroshot/cluewsc.zip)) | WSC Winograd模式挑战中文版 训练集（32），验证集（32），公开测试集（976），测试集（290），无标签语料（0） | 例子：    {"target":       {"span2_index": 37,       "span1_index": 5,       "span1_text": "床",       "span2_text": "它"},   "idx": 261,   "label": "false",   "text": "这时候放在床上枕头旁边的手机响了，我感到奇怪，因为欠费已被停机两个月，现在它突然响了。"}  "true"表示代词确实是指代span1_text中的名词的，"false"代表不是。  |

# 小样本学习
| **数据** | **描述 & 数据格式** |
| --- | --- |
| EPRSTMT ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/eprstmt.zip)) | 同上 |
| CSLDCP ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/csldcp.zip)) | 同上 |
| TNEWS ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/tnews.zip)) | 同上 |
| IFLYTEK ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/iflytek.zip)) | 同上 |
| OCNLI ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/ocnli.zip)) | 同上 |
| BUSTM ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/bustm.zip)) | 同上 |
| ChID ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/chid.zip)) | 同上 |
| CSL ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/csl.zip)) | 同上 |
| CLUEWSC ([直接下载](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue_fewshot/cluewsc.zip)) | 同上 |

# 多模态-检索数据
包括图文检索和文图检索

| **数据** | **描述** | **数据格式** |
| --- | --- | --- |
| Flickr8k-CN（[直接下载](https://github.com/li-xirong/flickr8kcn)） | **Flickr8k中文描述图文对** 每张图片对应5条文本描述：中文描述的翻译包含人工手写、人工翻译（仅test集）、机器翻译（百度翻译、谷歌翻译）**数据量: ** Pairs: 30000/5000/5000 (Images 6k, Text 30k) | caption：txt格式，包含原Flickr8k中image对应的id及不同翻译或手写版本的中文文本描述 image：jpg格式，以id区分，不同split的image id列表保存在txt中 |
| Flickr30k-CN（[直接下载](http://lixirong.net/data/mm2017/flickr30k-cn.tar.gz)） | **Flickr30k机器翻译文图对** 采用机器翻译原Flickr30k的描述（测试集为人工翻译），每张图片对应5条描述 **数据量: ** Pairs: 148915/5000/5000 (Images 29783, Text 148915) | caption：txt格式，包含原Flickr30k中image对应的id及机器翻译后的中文caption image：jpg格式，以id区分，不同split的image id列表保存在txt中 |
| COCO_CN（[Github](https://github.com/li-xirong/coco-cn)，需向原作者提交申请，通过后方可下载） | **MSCOCO人工翻译** 每张图片1-2条描述：中文描述的翻译包含人工手写、人工翻译（仅test集）、机器翻译（百度翻译）数据重新划分，与原MSCOCO不同 **数据量** Pairs: 20065/1000/1053 (Images: 18341,Text: 20065) | caption：txt格式，包含原MSCOCO2014中image的原id及人工翻译对应的中文caption image：jpg格式，以id区分，不同split的image id列表保存在txt中 |
| MUGE-Retrieval（[天池数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=107332)，需申请） | **电商领域文到图检索** 训练集1条query对应1个image，训练&验证集每条query对应5-6 images **数据量** Pairs: 248786/29806/30399 (Images: 129380, Text: 248786) | query：jsonl格式，每行一条query数据，包含query_id，query_text和对应image的id列表 image：tsv格式，每行一条image数据，包含image_id和image的base64编码 |
| AIC-ICC（[AI Challenger比赛官方网址]()） | **AI challenger比赛数据集**，包括Image captioning、关键点检测和机器翻译3个任务 每个图片对应5个描述 训练集30w图片，150w描述 验证集3w图片，15w描述 做图文检索任务时，重新划分过训练集验证集 Images：210000/30000/30000 Texts: 1050000/150000/300000 | caption：json格式，文件中每个样本包括url，image_id和5条captions image：jpg格式，以image_id命名 |
| ChineseFoodNet（[官方地址](https://sites.google.com/view/chinesefoodnet/)查看数据集并下载） | **中国食物数据集** 覆盖208个种类，185628张图片 **数据量** Images:145066/20254/20310 Text: 208 | 适合图片分类 |

# 多模态-生成数据
包括图到文，文到图数据

| **数据** | **描述** | **数据格式** |
| --- | --- | --- |
| Flickr30k-CN | 同上 | 同上 |
| COCO_CN | 同上 | 同上 |
| AIC-ICC | 同上 **适合文到图生成和图到文生成** | 同上 |
| MSCOCO_CN（[英文版下载链接](https://cocodataset.org/#download)，机器翻译的中文版本未公开） | **适合文到图生成和图到文生成** MSCOCO机器翻译文图对（2017版）,每张图片5条描述 Pairs: 591753/25014/- Image: 118287/5000/40671 Text:569002/24794/- | image: jpg格式。 text: json格式，key包括info, licenses, images, annotations。image-caption pairs 在annotations中 |
| MUGE-T2I（[天池数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=107332)，需申请） | **电商文到图生成** 每张图片对应一条描述 数据量: Pairs: 9w/5k/5k | image:  tsv格式,  \\t分隔： **图片id   \\t   商品图片内容 (base64编码）** text:   tsv格式，\\t分隔： **图片id   \\t    商品描述** |
| MUGE-IC（[天池数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=107332)，需申请） | **电商图到文生成** 每张图片对应一条描述 数据量: Pairs: 5w/5k/1w | image: tsv格式, \\t分隔： **img_id   \\t   img_content（base64编码）** caption: jsonl格式，key包括image_id，text |

# Acknowledge
以上数据收集自网上公开的数据，包括如下几个来源（如有侵权，烦请告知）：

- CLUE benchmark：[https://www.cluebenchmarks.com](https://www.cluebenchmarks.com/classification.html)
- CLUE datasets: [https://github.com/CLUEbenchmark/CLUEDatasetSearch](https://github.com/CLUEbenchmark/CLUEDatasetSearch/tree/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D)
- Wudao数据：[https://git.openi.org.cn/BAAI/WuDao-Data](https://git.openi.org.cn/BAAI/WuDao-Data)
- Wukong数据：[https://readpaper.com/paper/653639982984556544](https://readpaper.com/paper/653639982984556544)
- SophonPlus：[https://github.com/SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
- LAION 400M [https://laion.ai/laion-400-open-dataset/](https://laion.ai/laion-400-open-dataset/)
- LAION 5B [https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/](https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/)
