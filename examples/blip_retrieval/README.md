# Retrieval example with BLIP

Note that portions of models/ modelling and tokenizer code are adaptations of official repository of [BLIP](https://github.com/salesforce/BLIP).

### Image-Text Retrieval

#### Quick start
以Flickr30k为例
* 从官网下载Flickr30k完整数据集，保存到本地目录
* 在run_flickr_retrieval.sh文件中更改执行retrieval命令的image_root，并更改annotation下载路径ann_path
* 设置evaluate参数来判断是否需要对数据集进行微调，执行run_flickr_retrieval.sh文件
* 可以从下表link不同的pre-trained checkpoints来进行微调。当只对数据集进行验证时，推荐使用在Flickr30k微调后的checkpoint

|  checkpoints   | finetuned  | link  | 
|  ----  | :----:  | :----:  |
| BLIP w/ ViT-B  |     /      | [Download](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/blip/pretrained/model_base.pth) |
| BLIP w/ ViT-L  |     /      | [Download](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/blip/pretrained/model_large.pth) |
| BLIP w/ ViT-B  | Flickr30k  | [Download](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/blip/pretrained/model_base_retrieval_flickr.pth) |
| BLIP w/ ViT-L  | Flickr30k  | [Download](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/blip/pretrained/model_large_retrieval_flickr.pth) |

* 当采用ViT-L的模型进行验证及微调时，需要更改ViT及初始学习率等相关参数。对Flickr30k数据集，推荐的参数设置如下表所示：

|  ViT   | batch_size_train  | batch_size_test  | vit_ckpt_layer  | init_lr  | 
|  :----:  | :--:  |  :--:  | :--:  |  :----:  |
| base   | 32 |  64  |  4  |  1e-5 |
| large  | 16 |  32  |  10 |  5e-6 |

#### Dataset schema说明
* 训练/验证/测试的原图均存储在本地文件夹，通过参数image_root指定存储图像的本地路径
* image的文本描述文件以json格式存储在本地文件夹，通过指定参数ann_root说明annotation文件的存储路径
* 训练、验证和测试集合的图片描述应当存储在不同的文件中，并以‘train_file’，‘val_file’和‘test_file’指定不同划分的annotation文件名
* 对于Flickr30k，其annotation文件中的数据格式如：
`{"image": "flickr30k-images/1007129816.jpg", "caption": ["The man with pierced ears is wearing glasses and an orange hat.", "A man with glasses is wearing a beer can crocheted hat.", "A man with gauges and glasses is wearing a Blitz hat.", "A man in an orange hat starring at something.", "A man wears an orange hat and glasses."]}`

#### 自定义数据集
您可以基于上述Flickr30k的数据格式说明，使用自己的数据进行image-text retrieval任务微调及预测
* 修改data/retrieval_dataset.py文件自定义数据集的Dataset及DataLoader，也可以直接使用该数据格式定义（与Flickr30k一致）。当您使用自定义数据集时，输入参数'dataset'应被设置为'custom'
* 指定image_root，ann_root为自己的本地存储路径，同时修改'train_file'，'val_file'和'test_file'的文件名
* 选择合适的pre-trained checkpoint并修改对应参数（参见上文的推荐参数设置）
* Try it!


