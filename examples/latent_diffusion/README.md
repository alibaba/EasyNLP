# Tutorial of latent diffusion model
## 数据准备

1. 下载数据
```shell
if [ ! -f T2I_train.tsv ]; then
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/T2I_train.tsv
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/T2I_val.tsv
    wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/painter_text2image/T2I_test.tsv
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

MASTER_ADDR=localhost
MASTER_PORT=6027
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

1. 模型微调
```shell
  python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
    --mode=train \
    --worker_gpu=1 \
    --tables=T2I_train.tsv,T2I_val.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
    --learning_rate=4e-5 \
    --epoch_num=3 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=1000 \
    --sequence_length=288 \
    --micro_batch_size=16 \
    --app_name=latent_diffusion \
    --user_defined_parameters='
        pretrain_model_name_or_path=alibaba-pai/pai-diffusion-general-large-zh
        size=256
        text_len=32
        img_len=256
        reset_model_state_flag=True
      ' 
```

2. 模型预测
```shell
    python -m torch.distributed.launch $DISTRIBUTED_ARGS ./main.py \
      --mode=predict \
      --worker_gpu=1 \
      --tables=T2I_test.tsv \
      --input_schema=idx:str:1,text:str:1 \
      --output_schema=idx,text,gen_imgbase64 \
      --outputs=./tmp/T2I_outputs.tsv \
      --first_sequence=text \
      --checkpoint_dir=./tmp/finetune_model \
      --random_seed=42 \
      --logging_steps=100 \
      --save_checkpoint_steps=500 \
      --sequence_length=32 \
      --micro_batch_size=2 \
      --app_name=latent_diffusion \
      --user_defined_parameters="n_samples=2 write_image=True image_prefix=./output/" 
```