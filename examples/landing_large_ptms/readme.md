大模型在小样本数据上取得了不错的效果，但在很多实际场景中数据量不足的问题仍然制约着大模型的应用，如何提高预训练模型在小样本场景的泛化性还是个挑战。其次，大模型参数量太大导致训练和推理速度慢，严重影响到了需要较高QPS的线上场景，部署成本非常高，如何快速蒸馏出小模型也是个挑战。EasyNLP推出小样本学习功能，帮助用户在小样本场景快速训练一个效果好的模型落地。同时，EasyNLP支持知识蒸馏技术，可以将大模型压缩到小的高效的模型上线。

下面我们给出一个示例，将一个大的预训练模型（hfl/macbert-large-zh）在小样本场景上落地，并且蒸馏到小的模型上。如下所示，一个大模型（3亿参数）在一个小样本场景上原始的Accuracy为83.8%，通过小样本学习可以提升7%，达到90.6%。同时，如果用一个小模型（3百万参数）跑这个场景的话，效果仅有54.4%，可以把效果提升到75%（提升约21%），inference的时间相比大模型提升了约80倍。

|  | 模型 | 参数量 | Dev Set指标（Accuracy） | Batch Inference时间 |
| --- | --- | --- | --- | --- |
| 标准Finetune | hfl/macbert-large-zh | 325 Million | 83.75% | 3.22ms/sample (batch_size=8) |
| 标准Finetune | alibaba-pai/pai-bert-tiny-zh | 3 Million | 54.38% | 0.04ms/sample (batch_size=64) |
| 知识蒸馏Finetune | alibaba-pai/pai-bert-tiny-zh | 3 Million | 75.21% | 0.04ms/sample (batch_size=64) |
| 小样本Finetune | hfl/macbert-large-zh | 325 Million | 90.63% | 3.21ms/sample (batch_size=8) |

详细代码示例如下。
# 代码示例
## 数据准备
```bash
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/train.csv
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/landing_plm/dev.csv
```

## 小样本学习测试脚本
```python
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./fewshot_model/ \
    --learning_rate=1e-5 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=512 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
        enable_fewshot=True
        label_desc=好,差
        type=pet_fewshot
        pattern=text,是一条商品,label,评。
    "
```
## 知识蒸馏测试脚本
```python
# train teacher
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./teacher_model/ \
    --learning_rate=1e-5 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
    "

# data augmentation
easynlp \
    --app_name=data_augmentation \
    --worker_count=1 \
    --worker_gpu=1 \
    --mode=predict \
    --tables=train.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --outputs=aug.csv \
    --output_schema=augmented_data \
    --checkpoint_dir=_ \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=hfl/macbert-large-zh
        type=mlm_da
        expansion_rate=10
        mask_proportion=0.25
        remove_blanks=True
    "
    
# forward teacher logits
easynlp \
    --mode=predict \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=aug.csv \
    --outputs=logits.csv \
    --input_schema=text:str:1,label:str:1 \
    --output_schema=logits \
    --first_sequence=text \
    --checkpoint_path=./teacher_model/ \
    --micro_batch_size=8 \
    --sequence_length=128 \
    --app_name=text_classify

# train student w/ KD
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=aug.csv,dev.csv \
    --input_schema=text:str:1,label:str:1,logits:float:2 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./student_model/ \
    --learning_rate=1e-4 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=alibaba-pai/pai-bert-tiny-zh
        enable_distillation=True
        type=vanilla_kd
        logits_name=logits
        logits_saved_path=logits.csv
        temperature=1
        alpha=0.5
    "

# train student w/o. KD
easynlp \
    --app_name=text_classify \
    --mode=train \
    --worker_count=1 \
    --worker_gpu=1 \
    --tables=train.csv,dev.csv \
    --input_schema=text:str:1,label:str:1 \
    --first_sequence=text \
    --label_name=label \
    --label_enumerate_values=Positive,Negative \
    --checkpoint_dir=./small_model_2/ \
    --learning_rate=1e-4 \
    --epoch_num=5 \
    --random_seed=42 \
    --save_checkpoint_steps=100 \
    --sequence_length=128 \
    --micro_batch_size=8 \
    --user_defined_parameters="
        pretrain_model_name_or_path=alibaba-pai/pai-bert-tiny-zh
"
```
