# Tutorial of Text to Image Generation Models
## 数据准备

1. 下载数据
```shell
if [ ! -f ./tmp/MUGE_train_text_imgbase64.tsv ]; then
    wget  -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/MUGE_train_text_imgbase64.tsv
    wget  -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/MUGE_val_text_imgbase64.tsv
    wget  -P ./tmp https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/MUGE_test.text.tsv
fi
```

2. 数据格式
数据是以 \t 分隔的 .txt 文件，包含三个字段：idx, text, imgbase64

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

1. 模型微调
```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
    --mode=train \
    --worker_gpu=1 \
    --tables=./tmp/MUGE_train_text_imgbase64.tsv,./tmp/MUGE_val_text_imgbase64.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
    --learning_rate=4e-5 \
    --epoch_num=40 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=1000 \
    --sequence_length=288 \
    --micro_batch_size=16 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=alibaba-pai/pai-painter-base-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 
```

3. 模型预测
```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
    --mode=predict \
    --worker_gpu=1 \
    --tables=./tmp/MUGE_test.text.tsv \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs.tsv \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
    --sequence_length=288 \
    --micro_batch_size=16 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
        max_generated_num=1
      '
```