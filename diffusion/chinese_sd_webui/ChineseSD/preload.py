import os
from modules import paths

def preload(parser):
    parser.add_argument("--chinese-diffusion-dir-master", type=str, help="Path to directory with all chinese diffusion models. Contains chinese base diffusion ,chinese controlnet models,chinese lora models and so on", default=None)
    parser.add_argument("--chinese-diffusion-dir", type=str, help="Path to directory with chinese base diffusion models.", default=None)
    parser.add_argument("--chinese-controlnet-dir", type=str, help="Path to directory with chinese controlnet models.", default=None)
    parser.add_argument("--chinese-lora-dir", type=str, help="Path to directory with chinese lora models.", default=None)