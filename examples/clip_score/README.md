# CLIP score example

### 准备工作
* 安装好EasyNLP
* 进入目录 ./examples/clip_score

### 数据格式
数据格式为制表符分隔的两列 文本\t图片base64编码
相似度计算权重使用 wukong_vit_l_14_clip

### 执行
执行命令bash clip_score.sh

示例输出
```
pair number:  torch.Size([40])
tensor([0.0919, 0.1189, 0.1263, 0.0369, 0.1534, 0.1141, 0.1171, 0.0958, 0.1314,
        0.1175, 0.0507, 0.0593, 0.1364, 0.1074, 0.0397, 0.0367, 0.1115, 0.1431,
        0.1071, 0.0965, 0.1604, 0.1243, 0.1024, 0.0681, 0.1220, 0.1261, 0.1290,
        0.1068, 0.1126, 0.0381, 0.1011, 0.1086, 0.1263, 0.1108, 0.0933, 0.1422,
        0.1177, 0.0586, 0.0702, 0.1366], device='cuda:0')
averaged consine similarity  tensor(0.1037, device='cuda:0')
```

### 自定义开发
* 修改相似度计算逻辑, 请参考EasyNLP/easynlp/appzoo/wukong_clip/evaluator.py, 修改完成后记得python setup.py install对修改版本进行安装以生效


### 注意事项
* 对比学习得到的模型会使 ground truth 的余弦相似度相对于负例的余弦相似度排名更靠前，但是余弦相似度的具体数值不一定很大
