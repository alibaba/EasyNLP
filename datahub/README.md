EasyNLP提供常见的中文数据集的下载，同时提供脚本处理中文数据（待发布），敬请关注。
# 通用NLU任务
| **数据** | **描述** | **备注** |
| --- | --- | --- |
| AFQMC ([链接](https://www.cluebenchmarks.com/introduce.html)) | 蚂蚁金融语义相似度, 数据量：训练集（34334）验证集（4316）测试集（3861) | json格式，包括句子1，句子2，和标签，样例：{"sentence1": "xxx", "sentence2": "xxx", "label": "0"} |
| TNEWS1.1 ([链接](https://github.com/skdjfla/toutiao-text-classfication-dataset)) | 今日头条中文新闻（短文本）分类, 数据量：训练集(53,360)，验证集(10,000)，测试集(10,000) | json格式，包括分类ID，分类名称，和新闻文本，样例：{"label": "102", "label_des": "news_entertainment", "sentence": "xxx"} |
| IFLYTEK ([链接](https://www.cluebenchmarks.com/introduce.html)) | 长文本分类, 数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)  |  |
| OCNLI_50K ([链接](https://www.cluebenchmarks.com/introduce.html)) |  中文自然语言推理, 数据量：训练集(50000)，验证集(3000)，测试集(3000)  |  |
| WSC1.1([链接](https://www.cluebenchmarks.com/introduce.html)) |  代词消歧 （文本分类）, 数据量：训练集(1000)，验证集(300)，测试集(300)  |  |
| CSL([链接](https://www.cluebenchmarks.com/introduce.html)) |  论文关键词识别（文本分类）, 数据量：训练集(20,000)，验证集(3,000)，测试集(3,000)  |  |
| CMRC2018([链接](https://www.cluebenchmarks.com/introduce.html)) |  简体中文阅读理解任务, 数据量：训练集(短文数2,403，问题数10,142)，验证集(短文数256，问题数1,002)，测试集(短文数848，问题数3,219)  |  |
| CHID1.1([链接](https://www.cluebenchmarks.com/introduce.html)) |  成语阅读理解填空, 数据量：训练集(84,709)，验证集(3,218)，测试集(3,231)  |  |
| C3_1.1([链接](https://www.cluebenchmarks.com/introduce.html)) | 中文多选阅读理解, 数据量：训练集(11,869)，验证集(3,816)，测试集(3,892)  |  |

# 中文文本分类
| **数据** | **描述** | **备注** |
| --- | --- | --- |
| TNEWS1.1 | 详见通用NLU任务 |  |
| IFLYTEK | 详见通用NLU任务 |  |
| AFQMC | 详见通用NLU任务 |  |
| WSC1.1 | 详见通用NLU任务 |  |
| CSL | 详见通用NLU任务 |  |
| tc-corpus-answer([链接](http://www.nlpir.org/wordpress/2017/10/02/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E8%AF%AD%E6%96%99%E5%BA%93%EF%BC%88%E5%A4%8D%E6%97%A6%EF%BC%89%E6%B5%8B%E8%AF%95%E8%AF%AD%E6%96%99/)) | 复旦大学计算机信息与技术系国际数据库中心自然语言处理小组，训练9804篇（train），测试9833篇（answer），标签为20个类别 | 压缩包，包括train.rar, answer.rar |
| kesci-shorttext([链接](https://www.kesci.com/home/dataset/5dd645fca0cb22002c94e65d/files)) | 数据集来源于Kesci平台，为新闻标题领域短文本分类任务。内容大多为短文本标题(length<50)，数据包含15个类别，共38w条样本 |  |
| Sogou-CA([链接](http://www.sogou.com/labs/resource/ca.php)) | 数据来自若干新闻站点2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据 |  |
| Sogou-CS([链接](http://www.sogou.com/labs/resource/cs.php)) | 数据来源为搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据 |  |
| ChnSentiCorp_htl_all([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb)) | 7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论 |  |
| waimai_10k([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/waimai_10k/intro.ipynb)) | 某外卖平台收集的用户评价，正向 4000 条，负向 约 8000 条 |  |
| online_shopping_10_cats([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb)) | 10 个类别，共 6 万多条评论数据，正、负向评论各约 3 万条，包括书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店 |  |
| weibo_senti_100k([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb)) | 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条 |  |
| simplifyweibo_4_moods([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb)) | 36 万多条，带情感标注 新浪微博，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条 |  |
| dmsc_v2([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dmsc_v2/intro.ipynb)) | 28 部电影，超 70 万 用户，超 200 万条 评分/评论 数据 |  |
| yf_dianping([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_dianping/intro.ipynb)) | 24 万家餐馆，54 万用户，440 万条评论/评分数据 |  |
| yf_amazon([链接](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/yf_amazon/intro.ipynb)) | 52 万件商品，1100 多个类目，142 万用户，720 万条评论/评分数据 |  |

# 中文文本摘要任务
以下数据大部分来自[liucongg/GPT2-NewsTitle](https://github.com/liucongg/GPT2-NewsTitle)收集整理
| **数据** | **描述** | **备注** |
| --- | --- | --- |
| 清华新闻数据 [(百度云盘](https://pan.baidu.com/s/1a-CUtTc5xQFB9_EJaxDklA) 提取码： vhol |标题生成任务（短文本生成式摘要)清华新闻（THUCNews）数据是清华大学自然语言处理实验室整理，根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成.利用其正文与标题，可以构成新闻标题生成数据总数量：830749个样本；标题：平均字数 19； 正文：平均字数 892 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| 搜狗新闻数据 [百度云盘](https://pan.baidu.com/s/1vgfa5gnIHTYpoYptuHo6gQ) 提取码：ode6 | 标题生成任务（短文本生成式摘要)搜狗新闻（SogouCS）数据是搜狗实验室整理，来自搜狐新闻2012年6月—7月利用其正文与标题，可以构成新闻标题生成数据。整理后数据信息如下：总数量：1245835个样本；标题：平均字数 17 正文：平均字数 494 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| nlpcc2017摘要数据 [百度云盘](https://pan.baidu.com/s/1v7QFJ3hl_ALb2DEEq0umRQ) 提取码：e0zq  | 标题生成任务（短文本生成式摘要) 神策杯2018摘要数据是“神策杯”2018高校算法大师赛的比赛数据 总数量：108089个样本；摘要：平均字数 24 正文：平均字数 1055 | son格式包括title，content.其中content新闻正文 title为新闻的标题 |
| csl摘要数据 [百度云盘](https://pan.baidu.com/s/1qrzhsWq8SGQ1-W8VizSY9w) 提取码：0qot  | 计算机领域的论文摘要和标题数据，可用于短文本摘要生成。总数量：3500个样本；标题：平均字数 18，正文：平均字数 200，字数标准差 63，最大字数 631，最小数字 41  | json格式包括title，content.其中content为科研论文的摘要 title为科研论文的标题 |
| 教育培训行业摘要数据 [百度云盘](https://pan.baidu.com/s/1sjOkp8LKGVmY6h0QXl5m7g) 提取码：kjz3  | 标题生成任务（短文本生成式摘要）教育培训行业摘要数据是github作者wonderfulsuccess整理，数据主要由教育培训行业主流垂直媒体的历史文章 总数量：24423个样本；摘要：平均字数 52 正文：平均字数 2016 | json格式包括title，content.其中content新闻正文title为新闻的标题 |
| lcsts摘要数据 [百度云盘](https://pan.baidu.com/s/1J2NcMfxpGGG_BG1Wx0lHGA) 提取码：bzov | 标题生成任务（短文本生成式摘要) lcsts摘要数据是哈尔滨工业大学整理，基于新闻媒体在微博上发布的新闻摘要创建了该数据集 总数量：2108915个样本；摘要：平均字数 18 正文：平均字数 104 | json格式包括title，content.其中content新闻正文title为新闻的标题 |
| 神策杯2018摘要数据 [百度云盘](https://pan.baidu.com/s/1WFimCGk6y-nfSdPRbCrV8Q) 提取码：6f4f | 标题生成任务（短文本生成式摘要) 神策杯2018摘要数据是“神策杯”2018高校算法大师赛的比赛数据 总数量：108089个样本；摘要：平均字数 24 正文：平均字数 1055 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
| 万方摘要数据 [百度云盘](https://pan.baidu.com/s/1RFNFagKnxf2JKnjwBDecPA) 提取码： p69g | 标题生成任务（短文本生成式摘要）万方摘要数据是github作者EachenKuang整理，数据是从万方数据库爬取的文献摘要数据 总数量：3590个样本；摘要（论文标题）：平均字数 30 正文（论文摘要）：平均字数 295 | json格式包括title，content.其中content为科研论文的摘要 title为科研论文的标题 |
| 微信公众号摘要数据 [百度云盘](https://pan.baidu.com/s/1OBn8kyZEsUeiV_kw4OJYnQ) 提取码： 5has | 标题生成任务（短文本生成式摘要) 微信公众号摘要数据是github作者nonamestreet整理。整理后数据信息如下：总数量：712826个样本；标题：平均字数 22 正文：平均字数 1499 | json格式包括title，content.其中content新闻正文title为新闻的标题 |
| news2016zh新闻数据 [百度云盘](https://pan.baidu.com/s/1S3YhetbEZuSfYbfSLeRfSg) 提取码： qsj1 | 标题生成任务（短文本生成式摘要) news2016zh新闻数据是 CLUEbenchmark整理 总数量：2317427个样本；标题：平均字数 20 正文：平均字数 1250 | json格式包括title，content.其中content新闻正文 title为新闻的标题 |
|CLTS-Dataset  [百度云盘](https://pan.baidu.com/s/1skhl1HKUfRyFa7z3t8dH-g)提取码：请联系liuxiaojun@iie.ac.cn|文本摘要任务 CLTS 是一个新的中文长文本摘要数据集，提取自中文新闻网站 ThePaper.cn。生成的数据集版本包含超过 180,000 个长序列对，其中每篇文章由多个段落组成，每个摘要由多个句子组成。| |
|CN-Fin [TaskSumm](https://github.com/TangMoming/TaskSumm)|工业场景，金融文本摘要数据集||

# 其他数据（TODO）
# Acknowledge
以上数据收集自网上公开的数据，包括如下几个来源（如有侵权，烦请告知）：

- CLUE benchmark：[https://www.cluebenchmarks.com](https://www.cluebenchmarks.com/classification.html)
- CLUE datasets: [https://github.com/CLUEbenchmark/CLUEDatasetSearch](https://github.com/CLUEbenchmark/CLUEDatasetSearch/tree/master/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D)
- Wudao数据：[https://git.openi.org.cn/BAAI/WuDao-Data](https://git.openi.org.cn/BAAI/WuDao-Data)
- Wukong数据：[https://readpaper.com/paper/653639982984556544](https://readpaper.com/paper/653639982984556544)
- SophonPlus：[https://github.com/SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
