# DiffSynth

[arXiv](https://arxiv.org/abs/2308.03463) | [Source Code](https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth) | [Project Page](https://anonymous456852.github.io/)

DiffSynth is an open-source project that aims to apply diffusion models to video synthesis. You can use DiffSynth to synthesize coherent and realistic videos.

**Now an extention of stable-diffusion-webui is available! See [here](https://github.com/Artiprocher/sd-webui-fastblend).** This extension is an implementation of the fast blending algorithm in DiffSynth. We notice that this algorithm is very effective. Thus we develop this extension independently, making it easy to use.

## Installation

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
    - diffusers==0.18.0
    - safetensors
    - imageio
    - scipy
    - scikit-image
    - controlnet_aux==0.0.5
    - cupy
```

```shell
conda env create -f environment.yml
conda activate DiffSynth
```

Note that some components are implemented using cupy. If you find cupy doesn't work, maybe you need to reinstall it manually depending on your CUDA, see [this document](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi) for more details.

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

Please refer to our [project page](https://anonymous456852.github.io/) to see examples synthesized by DiffSynth. The config template of each task is in `./config/`, and we may update these templates in future.

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

DiffSynth is still under development. Now we recommend you to only use `PySynthSmoother`, although we have provided other smoothers in `./DiffSynth/smoother`. The other smoothers may be detated in future.

`PySynthSmoother` is a deflickering algorithm based on [Ebsynth](https://github.com/jamriska/ebsynth). It supports two modes:

```json
"smoother": "PySynthSmoother",
"smoother_config": {
    "speed": "fastest",
    "window_size": 30
}
```
or
```json
"smoother": "PySynthSmoother",
"smoother_config": {
    "speed": "slowest",
    "window_size": 30
}
```

This algorithm will blend the frames in a sliding window. It may make the video foggy when window size is large.

If `speed` is `fastest`, the time complexity is O(nlogn), where n is the number of frames.

If `speed` is `slowest`, the time complexity is O(nk), where k is the size of sliding window.

Additionally, you can adjust the contrast and sharpness in the smoother. You only need to add the following parameters in the `smoother_config`.

```json
"postprocessing": {
    "contrast": 1.5,
    "sharpness": 5.0
}
```

* `post_smoother` and `post_smoother_config`

If you want to apply the smoother again after the denoising process, please add `post_smoother` and `post_smoother_config` to the config file. Empirically, using `PySynthSmoother` as `post_smoother` with a small sliding window can make the video looks better.

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
