# Hugging Face Code adapt to EasyNLP

#### 运行步骤
1. 安装EasyNLP
2. 使用EasyNLP命令启动main.py
3. 对应train和predict运行脚本如下
```

easynlp \
    --mode=train \
    --tables=textclassify_data/train.tsv,textclassify_data/dev.tsv \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --label_enumerate_values=0,1 \
    --checkpoint_dir=./textClassify_checkpoint_model/ \
    --learning_rate=1e-5 \
    --epoch_num=1 \
    --logging_steps=100 \
    --random_seed=42 \
    --save_checkpoint_steps=50 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters=pretrain_model_name_or_path=bert-base-uncased \
```
```
easynlp \
    --mode=predict \
    --tables=textclassify_data/dev.tsv \
    --outputs=pred.tsv \
    --output_schema=predictions,probabilities,logits,output \
    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
    --first_sequence=sent1 \
    --second_sequence=sent2 \
    --label_name=label \
    --append_cols=sid1,label \
    --checkpoint_dir=./textClassify_checkpoint_model/ \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters=None \
```

#### Hugging Face 用户需要注意的地方

1. main.py中损失函数需要根据自己的任务定义，默认是Cross-Entropy损失计算，损失函数是动态绑定到model实例化上
2. Train状态下，只需要将model实例化改成用hugging face的transformers库初始化即可，EasyNLP可以兼容Hugging Face模型
3. Predict状态下，对于原本EasyNLP改动比较大，Hugging Face用户可以通过改动Predictor中preprocess/postprocess/run来对任务进行适配