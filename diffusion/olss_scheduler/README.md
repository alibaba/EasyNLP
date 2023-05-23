# OLSS (Optimal Linear Subspace Search)

In this project, we propose a new diffusion scheduler called OLSS. Given a few examples, OLSS can searching for the optimal approximation process of the complete generation process. After searching, OLSS is able to generate high-quality images with a very small number of steps.

在本项目中，我们提出了一个新的名为 OLSS 的 diffusion scheduler. 给定几个例子，OLSS 可以搜索完整生成过程的最优近似过程。在搜索完毕后，OLSS 能够以极少的步数生成高质量的图片。

## Examples 样例

Prompt: 亭台楼阁，曲径通幽，水墨绘画，中国风

Seed: 0

|OLSS (10 steps)|DPM-Solver++ (10 steps)|
|-|-|
|![](images/Building_OLSS_0.png)|![](images/Building_DPMSolver_0.png)|

Prompt: 层峦叠嶂，水墨绘画，中国风

Seed: 0

|OLSS (10 steps)|DPM-Solver++ (10 steps)|
|-|-|
|![](images/Mountain_OLSS_0.png)|![](images/Mountain_DPMSolver_0.png)|

## Usage 使用

```
pip install diffusers torch
```

We provide a demo here:

我们在这里提供了一个演示程序：

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler
from olss import SchedulerWrapper


# In this demo, we use our Chinese artist model. You can load any other models if you want.
pipe = StableDiffusionPipeline.from_pretrained("alibaba-pai/pai-diffusion-artist-large-zh")
pipe = pipe.to("cuda")

# Build an OLSS scheduler. This scheduler requires a reference scheduler. We use DDIM here.
pipe.scheduler = SchedulerWrapper(DDIMScheduler.from_config(pipe.scheduler.config))

# Before we generate images using OLSS scheduler, generate some examples to train it.
train_steps = 100
inference_steps = 10
prompt = "亭台楼阁，曲径通幽，水墨绘画，中国风"
for i in range(9):
    image = pipe(prompt=prompt, num_inference_steps=train_steps)["images"][0]
pipe.scheduler.prepare_olss(inference_steps)

# Generate some images using our OLSS scheduler. We can also generate images with other
# prompts. Using similar prompts for training is highly recommended, because OLSS can learn
# to construct a fast and high-quality generation process by analyzing given examples.
prompt = "亭台楼阁，曲径通幽，水墨绘画，中国风"
image = pipe(prompt=prompt, num_inference_steps=inference_steps)["images"][0]
image.save(f"{prompt}.png")
```

