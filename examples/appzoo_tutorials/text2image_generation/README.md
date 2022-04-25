# Tutorial of Text to Image Generation Models
## 数据准备

1. 下载数据
if [ ! -f ./tmp/T2I_train.txt ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_train.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_val.txt
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/artist_text2image/T2I_test.txt
    mkdir tmp/
    mv *.txt tmp/
fi

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
1. 预训练

```shell
if [ ! -f ./tmp/vqgan_f16_16384.bin ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/vqgan_f16_16384.bin
    mv vqgan_f16_16384.bin tmp/
  fi

  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=train \
    --tables=./tmp/T2I_train.txt,./tmp/T2I_val.txt \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/artist_model_pretrain \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        vqgan_ckpt_path=./tmp/vqgan_f16_16384.bin
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 
```

2. 模型微调
```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=train \
    --tables=./tmp/T2I_train.txt,./tmp/T2I_val.txt \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
    --learning_rate=4e-5 \
    --epoch_num=1 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=200 \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=artist-base-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 
```

3. 模型预测
```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/appzoo_tutorials/text2image_generation/main.py \
    --mode=predict \
    --tables=./tmp/T2I_test.txt \
    --input_schema=idx:str:1,text:str:1 \
    --first_sequence=text \
    --outputs=./tmp/T2I_outputs.txt \
    --output_schema=idx,text,gen_imgbase64 \
    --checkpoint_dir=./tmp/artist_model_finetune \
    --sequence_length=288 \
    --micro_batch_size=8 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      '
```