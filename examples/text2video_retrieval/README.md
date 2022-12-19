# Retrieval video with CLIP

### 准备工作
* 安装好EasyNLP
* 进入目录 ./examples/text2video_retrieval
* 下载并生成数据 sh preprocess_video_frame.sh

### 数据格式
对于train与evaluate 数据格式为制表符分隔的两列 文本\t视频提取帧存放路径
对于predict 数据格式为单列 文本 或 视频提取帧存放路径

### Train
执行命令 sh run_clip_local_appzoo.sh 0 train_en
其中0是所用显卡编号，含义同CUDA_VISIBLE_DEVICES=0

### evaluate
执行命令 sh run_clip_local_appzoo.sh 0 evaluate_en

### predict
predict用于生成测试数据的CLIP特征
sh run_clip_local_appzoo.sh 0 predict_en_text
默认将生成测试文本的特征，修改run_clip_local_appzoo.sh文件中的代码可生成测试video的特征

### 自定义开发
* batch size 和 learning rate等参数在run_clip_local_appzoo.sh文件中修改
* 修改模型底层逻辑如dataset,loss,evaluator等, 请参考easynlp/appzoo/text2video_retrieval 和 easynlp/modelzoo/models/clip 这两个文件夹, 修改完成后记得python setup.py install对修改版本进行安装以生效.
