# Tutorial of Text to Image Generation Models
## 数据准备
1. 数据格式
数据是以 \t 分隔的 .txt 文件，包含三个字段：idx, text, imgbase64
2. Prepare data
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
1. 下载vqgan模型[vqgan_imagenet_f16_16384](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=/ckpts/last.ckpt&dl=1)
2. 执行训练脚本
examples/appzoo_tutorials/text2image_generation/run_user_defined_local.sh

```python
python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
  --mode train \
  --tables=train.txt,val.txt \
  --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
  --first_sequence=text \
  --second_sequence=imgbase64 \
  --checkpoint_dir=/tmp/artist_model \
  --learning_rate=4e-5 \
  --epoch_num=3 \
  --random_seed=42 \
  --logging_steps=100 \
  --save_checkpoint_steps=500 \
  --sequence_length=288 \
  --micro_batch_size=8 \
  --app_name=text2image_generation \
  --user_defined_parameters='
      tokenizer_name_or_path=hfl/chinese-roberta-wwm-ext
      size=256
      text_len=32
      img_len=256
      img_vocab_size=16384
      vqgan_ckpt_path=/root/imagenet_f16_16384/checkpoints/last.ckpt
    ' 
```