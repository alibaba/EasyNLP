# Retrieval example with CLIP

### 准备工作
* 安装好EasyNLP
* 进入目录 ./examples/appzoo_tutorials/clip

### 数据格式
对于train与evaluate 数据格式为制表符分隔的两列 文本\t图片base64编码
对于predict 数据格式为单列 文本 或 图片base64编码

### Train
执行命令 sh run_clip_local_appzoo.sh train_cn

示例输出
```
Training Time: 45.49571466445923, rank 0, gsteps 32
127 280 334 500
[49.400000000000006, 25.4, 56.00000000000001, 66.8]
[2022-04-25 10:57:53,807 INFO] Inference time = 1.79s, [3.5709 ms / sample] 
[2022-04-25 10:57:53,808 INFO] Saving best model to ./clip_model/pytorch_model.bin...
[2022-04-25 10:58:14,532 INFO] Best score: 0.49400000000000005
[2022-04-25 10:58:14,534 INFO] Training Time: 113.8211259841919
[2022-04-25 10:58:14,535 INFO] Duration time: 167.77124881744385 s
```

### evaluate
执行命令 sh run_clip_local_appzoo.sh evaluate_cn

示例输出
```
143 295 360 500 # r1个数 r5个数 r10个数 总评估样本数
[53.19999999999999, 28.599999999999998, 59.0, 72.0] # [mean recall, recall@1, recall@5, recall@10]
[2022-04-25 11:00:09,800 INFO] Inference time = 1.77s, [3.5340 ms / sample] 
[2022-04-25 11:00:09,800 INFO] Duration time: 92.21781587600708 s
```

### predict
predict用于生成测试数据的CLIP特征
sh run_clip_local_appzoo.sh predict_cn_text
默认将生成测试文本的特征，修改run_clip_local_appzoo.sh文件中的代码可生成测试图片base64的特征

### 自定义开发
* batch size 和 learning rate等参数在run_clip_local_appzoo.sh文件中修改
* 修改模型底层逻辑如dataset,loss,evaluator等, 请参考easynlp/appzoo/clip 和 easynlp/modelzoo/models/clip 这两个文件夹, 修改完成后记得python setup.py install对修改版本进行安装以生效.
