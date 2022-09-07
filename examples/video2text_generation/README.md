# Tutorial of Text to Image Generation Models
## 数据准备

1. 下载数据
```shell
if [ ! -f ./tmp/IC_train_base64.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/image2text_generation/VC_train_base64.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/image2text_generation/VC_val_base64.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/image2text_generation/VC_test_base64.txt
    mv *.txt tmp/
fi
```

2. 数据格式
数据是以 \t 分隔的 .txt 文件，包含三个字段：idx, [imgbase64\_1, imgbase64\_2, ....., imgbase64\_frame\_num] , , text

3. 准备自己的数据
将image编码为base64形式：
```python
import base64
from io import BytesIO
from PIL import Image

img = Image.open(fn)
img_buffer = BytesIO()
img.save(img_buffer, format=img.format)
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data) # bytes
```

base64数据解码：
```python
import base64
from io import BytesIO
from PIL import Image
img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64)))
```

## 模型训练
1. 预训练

frame_num是视频分为的帧的个数。目前支持frame_num=1,2,3,....... img_len
如果没有指定，默认是前三帧。

```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=train \
    --tables=./tmp/VC_train_base64.txt,./tmp/VC_val_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_gen_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        frame_num=4
        vit_ckpt_path=ViT-L/14
        img_size=224
        img_len=256
        text_len=32
        pretrain_model_name_or_path=bert-base-chinese
        block_size=288
        n_layer=12
        n_head=12
        n_embd=768
      ' 
```

2. 模型微调
```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=train \
    --tables=./tmp/VC_train_base64.txt,./tmp/VC_val_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1,text:str:1 \
    --first_sequence=imgbase64 \
    --second_sequence=text \
    --checkpoint_dir=./tmp/v2t_gen_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        frame_num=4
        pretrain_model_name_or_path=./tmp/v2t_gen_model_pretrain
        img_size=224
        img_len=256
        text_len=32
      ' 
```

3. 模型预测
```shell
  rm -rf ./tmp/IC_outputs.txt
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/video2text_generation/main.py \
    --mode=predict \
    --tables=./tmp/VC_test_base64.txt \
    --input_schema=idx:str:1,imgbase64:str:1 \
    --first_sequence=imgbase64 \
    --outputs=./tmp/VC_outputs.txt \
    --output_schema=idx,gen_text \
    --checkpoint_dir=./tmp/v2t_gen_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=video2text_generation \
    --user_defined_parameters='
        frame_num=4
        img_size=224
        text_len=32
        img_len=256
        max_generated_num=4
      '
```
