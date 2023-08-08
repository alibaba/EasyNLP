# Diffusion Video Stylizer

**The new version is here**: https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth

We designed a deflicker algorithm for diffusion models. This algorithm is still under development and this released version is an initial version. Please stay tuned for future updates.

## Install

`environment.yaml`:

```yaml
name: video-stylizer
channels:
  - xformers
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.9.16
  - pip=23.0.1
  - cudatoolkit
  - pytorch
  - torchvision
  - xformers
  - pip:
    - transformers
    - accelerate
    - datasets
    - bitsandbytes
    - opencv-python-headless
    - einops
    - timm
    - diffusers
    - safetensors
```

```
conda env create -f environment.yaml
```

## Usage

We provide an example here.

```python
import torch, PIL
from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from DiffusionVideoStylizer.pipelines import VideoStylizingPipeline, ControlnetImageProcesserDepth, ControlnetImageProcesserHED


# load models in diffusers format
controlnet_depth = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.bfloat16)
controlnet_hed = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.bfloat16)
pipe = VideoStylizingPipeline.from_pretrained("gsdf/Counterfeit-V2.5", controlnet=[controlnet_depth, controlnet_hed], torch_dtype=torch.bfloat16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# load frames
height, width = 1280, 720
frames = [PIL.Image.open(f"your_video_frames/{i}.png").resize((width, height)) for i in range(0, 10)]
image_reference = PIL.Image.open("your_reference_image.png").resize((width, height))

# write your prompts here
prompt = "masterpiece, best quality, colorful, a girl is dancing"
negative_prompt = "extra fingers, fewer fingers, watermark"

# Go!
images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    frames=frames,
    image_reference=image_reference,
    image_processers=[
        ControlnetImageProcesserDepth(resolution=512),
        ControlnetImageProcesserHED(resolution=512)
    ],
    num_inference_steps=5,
    seed_of_noise=0,
    combine_pattern=["reference", -1, 0, 1],
    controlnet_scale=[0.6, 0.6],
    img2img_strength=0.6,
)["images"]

# save each frames
for i, image in enumerate(images):
    image.save(f"{i}.png")
```

Our algorithm is inspired by `multiframe`, and we improved it for better performance. In this algorithm, we generate a frame according to its adjusting frames and a reference image. You can modify this pattern by editing the parameters `combine_pattern`. For example, `["reference", -1, 0, 1]` represents that each frame is generated according to the reference image, the previous frame, and the next frame (Yes! We can obtain the information from the next frame before it is generated!). Adding more frames to the pattern will lead to smoother video but makes the program run slower. The order in the pattern doesn't matter because we don't concatenate frames just like in `multiframe`.

We only tested the program on A100-SXM-80GB. If the video resolution does not exceed 540P, 10GB of GPU memory is enough. If the video resolution is up to 1080P, you need at least 22GB of GPU memory.
