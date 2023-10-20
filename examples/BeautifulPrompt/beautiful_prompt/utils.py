import json
import random
import os

from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def set_seed(seed: int):
    """
    Sets seeds across package dependencies for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    no_decay_name_list=[
        "bias", "LayerNorm.weight"
    ]
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def init_sd_pipeline(device = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
    '''
    Initializes the Stable Diffusion pipeline

    Args:
        device: The device to put the loaded model.
        kwargs: Keyword arguments to be passed to the underlying DiffusionPipeline object.
            This can include any of the arguments accepted by the from_pretrained() method of the DiffusionPipeline class.

    Returns:
        A diffusion pipeline.
    
    Example:

    ```python
    >>> from beautiful_prompt.utils import init_sd_pipeline
    >>> sd_pipeline = init_sd_pipeline()
    >>> sd_pipeline.text2img(prompt, width=512, height=512, negative_prompt=neg_prompt, max_embeddings_multiples=3).images[0]
    '''
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    sd_pipeline = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        custom_pipeline="waifu-research-department/long-prompt-weighting-pipeline",
        safety_checker=None, # Comment it for working safely
        revision="fp16",
        torch_dtype=torch.float16,
        **kwargs
    )

    sd_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipeline.scheduler.config)
    sd_pipeline.set_progress_bar_config(disable=True)
    sd_pipeline.to(device)

    return sd_pipeline
