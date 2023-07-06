# EasyNLP PAI-Diffusion模型专区

阿里云机器学习PAI平台提供了一系列自研的Diffusion模型（特别是中文领域），并且开放了模型推理和各种微调功能。本文详细介绍推理和微调PAI-Diffusion模型的功能。使用方式分见如下列表：

- 使用PAI-Diffusion模型直接生成
- 全量微调PAI-Diffusion模型
- 使用LoRA轻量化微调PAI-Diffusion模型

# 使用PAI-Diffusion模型直接进行文图生成

### 环境搭建

```
pip install transformers accelerate datasets bitsandbytes torch torchvision diffusers
```

### 模型列表

使用PAI-Diffusion模型直接进行文图生成的方法与使用社区版Diffusion模型类似。我们提供了如下预制的PAI-Diffusion模型给您使用：

| 模型名                                      | 使用场景                                                     |
| ------------------------------------------- | ------------------------------------------------------------ |
| alibaba-pai/pai-diffusion-artist-large-zh  | 中文文图生成通用艺术模型，默认支持生成图像分辨率为512*512 |
| alibaba-pai/pai-diffusion-artist-large-zh-controlnet-canny   | 中文文图生成通用艺术模型的Canny ControlNet |
| alibaba-pai/pai-diffusion-artist-large-zh-controlnet-depth   | 中文文图生成通用艺术模型的Depth ControlNet |
| alibaba-pai/pai-diffusion-food-large-zh     | 中文美食文图生成，默认支持生成图像分辨率为512*512            |
| alibaba-pai/pai-diffusion-artist-xlarge-zh | 中文文图生成通用艺术模型（更大分辨率），默认支持生成图像分辨率为768*768 |

使用上述模型需遵守[AIGC模型开源特别条款](https://terms.alicdn.com/legal-agreement/terms/common_platform_service/20230505180457947/20230505180457947.html)。

### 文图生成

在PAI-DSW中，使用与微调社区Diffusion模型相同的环境配置，我们可以直接利用上述模型生成中文文本对应的图片，示例脚本如下：

```python
from diffusers import StableDiffusionPipeline

model_id = "alibaba-pai/pai-diffusion-artist-large-zh"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
prompt = "输入文本"
image = pipe(prompt).images[0]  
image.save("result.png")
```

如果需要使用其他模型，只需要替换对应model_id的值即可。                     

除了直接输入文本生成图像，PAI-Diffusion模型也支持其他从文本到图像生成的Pipeline。

### 文本引导的图像编辑

文本引导的图像编辑允许模型在给定输入文本和图像的基础上，生成相关的图像，示例脚本如下：

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

init_image = Image.open("image.png").convert("RGB")
init_image.thumbnail((512, 512))

model_id = "alibaba-pai/pai-diffusion-artist-large-zh"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
prompt = "输入文本"
image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
image.save("result.png")
```

如果需要使用其他模型，只需要替换对应model_id的值即可。

### 文本引导的局部重绘

使用 mask 遮盖部分区域，对图像中部分细节进行修改，示例脚本如下：

```python
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

image = Image.open("image.png")
mask_image = Image.open("mask_image.png")

model_id = "alibaba-pai/pai-diffusion-artist-large-zh"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
prompt = "输入文本"
image = pipe(prompt=prompt, image=image, mask_image=mask_image, strength=0.8).images[0]
image.save("result.png")
```

### 可控的图像生成

我们提供了两个 ControlNet 模型，您可以使用 ControlNet 模型进行可控的图像生成，示例脚本如下：

Canny ControlNet：

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2


def to_canny(image):
    low_threshold = 100
    high_threshold = 200
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


controlnet_id = "alibaba-pai/pai-diffusion-artist-large-zh-controlnet-canny"
controlnet = ControlNetModel.from_pretrained(controlnet_id)
model_id = "alibaba-pai/pai-diffusion-artist-large-zh"
pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet)
pipe = pipe.to("cuda")

image = Image.open("image.png")
controlnet_image = to_canny(image)
prompt = "输入文本"
image = pipe(prompt, controlnet_image).images[0]

controlnet_image.save("image_canny.png")
image.save("image_canny_output.png")
```

Depth ControlNet：

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2


def to_depth(image):
    depth_estimator = pipeline('depth-estimation')
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image


controlnet_id = "alibaba-pai/pai-diffusion-artist-large-zh-controlnet-depth"
controlnet = ControlNetModel.from_pretrained(controlnet_id)
model_id = "alibaba-pai/pai-diffusion-artist-large-zh"
pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet)
pipe = pipe.to("cuda")

image = Image.open("image.png")
controlnet_image = to_depth(image)
prompt = "输入文本"
image = pipe(prompt, controlnet_image).images[0]

controlnet_image.save("image_depth.png")
image.save("image_depth_output.png")
```

# 微调PAI-Diffusion模型

PAI-Diffusion模型可以使用我们提供的脚本 `diffusers_api/finetune.py` 进行微调，同样地，我们使用上一节微调社区Diffusion模型的开发环境运行脚本。微调PAI-Diffusion模型步骤如下：

### 步骤一：准备数据

准备训练数据集。本文使用某个中文文图对数据集的一个子集进行模型训练。训练数据集具体格式要求如下。

```
folder/train/image1.jpg
folder/train/image2.jpg
folder/train/image3.jpg
folder/train/metadata.jsonl
```

`metadata.jsonl`:

```
{"file_name": "image1.jpg", "text": "文本1"}
{"file_name": "image2.jpg", "text": "文本2"}
{"file_name": "image3.jpg", "text": "文本4"}
```

我们提供了一个示例数据集，供您下载：

```
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/pai_ldm_diffusers/dataset.tar
tar xvf dataset.tar
```

### 步骤二：构建Diffusion模型

设置 accelerate 的训练参数：

```python
accelerate config
```

训练命令示例如下：

```python
export MODEL_NAME="alibaba-pai/pai-diffusion-artist-large-zh"
export TRAIN_DIR="path_to_your_dataset"
export OUTPUT_DIR="path_to_save_model"

accelerate launch finetune.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR
```

其中，MODEL_NAME是用于微调的PAI-Diffusion模型名称，TRAIN_DIR是前述训练集的本地路径，OUTPUT_DIR为模型保存的本地路径。

### 步骤三：使用微调后的模型

当模型微调完毕之后同样可以使用如下示例代码进行文图生成：

```python
from diffusers import StableDiffusionPipeline

model_id = "path_to_save_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
prompt = "输入文本"
image = pipe(prompt).images[0]  
image.save("result.png")
```

其中 `path_to_save_model` 是前一步中保存模型的路径。


# 使用LoRA轻量化微调PAI-Diffusion模型

由于Diffusion类模型参数量庞大，直接微调这些模型会导致计算暴增。PAI-Diffusion模型可以使用LoRA（Low-Rank Adaptation）算法进行轻量化微调，大幅降低计算量。同样地，我们可以使用脚本 `diffusers_api/lora.py`，轻量化微调PAI-Diffusion模型进行文图生成，步骤如下：

### 步骤一：准备数据

本步骤和前一节相同。

### 步骤二：构建Diffusion模型

设置 accelerate 的训练参数：

```python
accelerate config
```

训练命令示例如下：

```python
export MODEL_NAME="alibaba-pai/pai-diffusion-artist-large-zh"
export TRAIN_DIR="path_to_your_dataset"
export OUTPUT_DIR="path_to_save_model"

accelerate launch lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR
```

其中，MODEL_NAME是用于微调的PAI-Diffusion模型名称，TRAIN_DIR是前述训练集的本地路径，OUTPUT_DIR为模型保存的本地路径。

### 步骤三：使用轻量化微调后的模型

当模型轻量化微调完毕之后同样可以使用如下示例代码进行文图生成：

```python
from diffusers import StableDiffusionPipeline

model_id = "path_to_save_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")
prompt = "输入文本"
image = pipe(prompt).images[0]  
image.save("result.png")
```

其中 `path_to_save_model` 是前一步中保存模型的路径。
