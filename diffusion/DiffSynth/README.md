# DiffSynth

[arXiv](https://arxiv.org/abs/2308.03463) | [Source Code](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth) | [Project Page](https://anonymous456852.github.io/)

DiffSynth is an open-source project that aims to apply diffusion models to video synthesis. You can use DiffSynth to synthesize coherent and realistic videos.

## Installation

### 1. Python Environment

environment.yml:

```yml
name: DiffSynth
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
    - imageio
    - scipy
    - scikit-image
    - controlnet_aux==0.0.5
```

```shell
conda env create -f environment.yaml
conda activate DiffSynth
```

### Compile Ebsynth NNF Estimator

We use [Ebsynth](https://github.com/jamriska/ebsynth) in the deflickering algorithm.

```shell
cd ebsynth
```

```shell
nvcc -arch compute_50 src/ebsynth_nnf.cpp src/ebsynth_cpu.cpp src/ebsynth_cuda.cu -I"include" -DNDEBUG -D__CORRECT_ISO_CPP11_MATH_H_PROTO -O6 -std=c++11 -w -Xcompiler -fopenmp -o ../bin/ebsynth_nnf
```

```shell
nvcc -arch compute_50 src/ebsynth.cpp src/ebsynth_cpu.cpp src/ebsynth_cuda.cu -I"include" -DNDEBUG -D__CORRECT_ISO_CPP11_MATH_H_PROTO -O6 -std=c++11 -w -Xcompiler -fopenmp -o ../bin/ebsynth
```

We only tested this component with CUDA 11.2/11.3 on Linux.

## Usage

```shell
python run_DiffSynth.py your_config_file.json
```

DiffSynth now supports the following five tasks:
1. Text-guided video stylization
2. Fashion video synthesis
3. Image-guided video stylization
4. Video Restoring
5. 3D rendering

Please refer to our [project page](https://anonymous456852.github.io/) to see examples synthesized by DiffSynth. The config template of each task is in `./config/`.

## Parameters

A config file consists of the following parameters:

### Base Diffusion Model

* `model_id`

```json
"model_id": "runwayml/stable-diffusion-v1-5",
```

The diffusion model you want to use. This project is developed based on diffusers, thus you can use a huggingface model id or a local path. We only use the tokenizer, text encoder, U-Net and VAE. The safety checker and scheduler will be ignored.

### ControlNet

* `controlnet_model_id`

```json
"controlnet_model_id": [
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_softedge"
]
```
Determine the ControlNet models you need. This project is developed based on diffusers, thus you can use a huggingface model id or a local path.

* `controlnet_processor`

```json
"controlnet_processor": [
    "ControlnetImageProcesserDepth",
    "ControlnetImageProcesserHED"
]
```
Determine the ControlNet processors you need. See the codes in `DiffSynth/controlnet_processors` for more details.

* `controlnet_scale`

```json
"controlnet_scale": [
    1.0,
    1.0
]
```
The conditioning scale of each ControlNet model.

* `style_image`

```json
"style_image": "style.png"
```
The input of Shuffle ControlNet.

### Smoother

* `smoother` and `smoother_config`

```json
"smoother": "EbsynthSmoother",
"smoother_config": {
    "bin_path": "bin/ebsynth",
    "cache_path": "cache",
    "smooth_index": [-3, -2, -1, 1, 2, 3]
}
```
or
```json
"smoother": "VideoPatchMatchSmoother",
"smoother_config": {
    "engine_name": "Ebsynth",
    "bin_path": "bin/ebsynth_nnf",
    "cache_path": "cache",
    "postprocessing": {
        "contrast": 1.5,
        "sharpness": 5.0
    }
}
```
The smoother contains the deflickering algorithm. It could be `EbsynthSmoother` or `VideoPatchMatchSmoother`, where the former is applied in a sliding window and the latter is applied in the whole video. Note that the video may be blurry when we use `VideoPatchMatchSmoother` to blend all frames. We use the parameter `postprocessing` to control the post-process the frames.

* `ignore_smoother_steps`

```json
"ignore_smoother_steps": 0
```
The smoother will be disabled in the last `ignore_smoother_steps` denoising steps. Sometimes the video looks better if we set `ignore_smoother_steps=1`.

* `smoother_interval`

```json
"smoother_interval": 5
```
Determine how frequent we use the smoother. For example, if `smoother_interval=5` and `num_inference_steps=20`, we use the smoother in the 1st, 6th, 11th and 16th frames.

* `extra_deflickering_range`

```json
"extra_deflickering_range": 60
```
We apply the deflickering algorithm to the video after all denoising steps. This parameter represent the size of sliding window of `EbsynthSmoother`. Leave it `0` if you want to skip this step.

### Video Input/Output

* `input_video`

```json
"input_video": "input_video.mp4"
```
The path of input video.

* `frame_height` and `frame_width`

```json
"frame_height": 512,
"frame_width": 960
```
The height and width of each frame. We will crop and resize each frame.

* `frame_interval`

```json
"frame_interval": 1
```
Determine the interval between two frames. For example, if `frame_interval=1`, we use every frame. If `frame_interval=2`, we use the 1st, 3rd, 5th, ... frames.

* `frame_maximum_count`

```json
"frame_maximum_count": 10
```
The maximum number of frames to be rerendered. You can use a very large value (e.g., 200) if you have enough memory. Don't be afraid, because the GPU memory required is not related to this parameter.

* `output_path`

```json
"output_path": "output_video"
```
The rendered video will be stored here.

### Parameters in Image Synthesis

* `prompt` and `negative_prompt`

```json
"prompt": "winter, snow, fishing rod, white, grass",
"negative_prompt": ""
```
Use prompts like in a image synthesis pipeline.

* `num_inference_steps`

```json
"num_inference_steps": 20
```
The number of denoising steps.

* `seed_of_noise`

```json
"seed_of_noise": 0
```
The random seed used for initialization.

* `img2img_strength`

```json
"img2img_strength": 1.0
```
The strength in an image-to-image pipeline. `img2img_strength=1.0` means the video is synthesized from scratch without any information from the input video, but you can use ControlNet models to control the content.

### Cross-Frame Attention

We modify the self-attention layers in UNet and ControlNet, making each frame interact with other frames using cross-frame attention.

* `frames_reference`

```json
"frames_reference": ["reference_frame_0.png", "reference_frame_1.png"],
```
The path of reference frames. Leave it empty if you don't want to use it.

* `combine_pattern`

```json
"combine_pattern": [-10000, "reference_0", -1, 0, 1]
```
Determine the receptive field of each frame. In this example, `0` represents this frame, `-1` represents the last frame, `1` represents the next frame, `-10000` represents the first frame (if the number of frames exceeds 10000), and `reference_0` represents the first reference frame. Please be careful because the we need to store all attention parameters of these frames on GPU.

## Tips

This algorithm may be slow while synthesizing long videos. Please be patient.
