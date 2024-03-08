# Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing  (CVPR 2024)
[arXiv Paper](https://arxiv.org/abs/2403.03431)

![teaser](assets/cases.png) ![other models](assets/other_models.png)


## Setup
This code was tested with Python 3.8 and [Pytorch](https://pytorch.org/) 1.11, using pre-trained models through [Hugging Face Diffusers](https://github.com/huggingface/diffusers#readme).
Specifically, we implemented our method over [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5). It can also be implemented over other stable diffusion models like [Realistic-V2](https://huggingface.co/SG161222/Realistic_Vision_V2.0), [Deliberate](https://huggingface.co/XpucT/Deliberate), and [Anything-V4](https://huggingface.co/xyn-ai/anything-v4.0).
When implemented over [Realistic-V2],[Deliberate] and [Anything-V4], you need to update diffusers to 0.21.1.
Additional required packages are listed in the requirements file.

### Creating a Conda Environment
```
conda env create -f environment.yaml
conda activate FPE
```

## Notebooks
[**edit_fake**][fake-edit] for Synthesis image editing.
[**edit_real**][real-edit] for Real image editing.
[**null_text_w_FPE**][null_text+FPE] Real image editing using NULL TEXT INVERSION to reconstucte image.
[**edit_fake**][fake-edit] using self attention map control in prompt-to-prompt code.

## Attention Control Options
 * `self_replace_steps`: specifies the fraction of steps to replace the self-attention maps.

## Gradio Running
```
cd gradio_app
python app.py --model_path "<path to your stable diffusion model>"
```
## Citation
``` bibtex
@misc{liu2024understanding,
      title={Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing},
      author={Bingyan Liu and Chengyu Wang and Tingfeng Cao and Kui Jia and Jun Huang},
      year={2024},
      eprint={2403.03431},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


[fake-edit]: edit_fake.ipynb
[real-edit]: edit_real.ipynb
[null_text+FPE]: null_text_w_FPE.ipynb
[FPE_in_p2p]: FPE_In_p2pcode.ipynb