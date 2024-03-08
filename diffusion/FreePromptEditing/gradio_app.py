import gradio as gr
import numpy as np
import torch
from diffusers import DDIMScheduler
import os
import argparse
import random
from typing import Optional
from torchvision.io import read_image
from Freeprompt.diffuser_utils import FreePromptPipeline
from Freeprompt.freeprompt import SelfAttentionControlEdit,AttentionStore
from Freeprompt.freeprompt_utils import register_attention_control_new
import torch.nn.functional as F
from scipy import ndimage


torch.set_grad_enabled(False)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device( "cpu")



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default = 'models/stable-diffusion-v1-5')


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:

    if seed is None:
        seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def consistent_synthesis(source_prompt, target_prompt, replace_scale,
                         replace_layers, image_resolution, ddim_steps, scale,
                         seed, appended_prompt, negative_prompt):

    seed_everything(seed)

    with torch.no_grad():
        if appended_prompt is not None:
            source_prompt += appended_prompt
            target_prompt += appended_prompt
        prompts = [source_prompt, target_prompt]

        # initialize the noise map
        start_code = torch.randn([1, 4, 64, 64], device=device)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        
        self_replace_steps = replace_scale
        NUM_DIFFUSION_STEPS = ddim_steps
        controller = SelfAttentionControlEdit(prompts, NUM_DIFFUSION_STEPS,replace_layers,self_replace_steps=self_replace_steps)
        register_attention_control_new(model, controller)

        # inference the synthesized image
        image_results = model(prompts, latents=start_code, guidance_scale=7.5,)
        image_results = image_results.cpu().permute(0, 2, 3, 1).numpy()

    return [image_results[0],
            image_results[1]]  # source, fixed seed, Editing

def create_demo_synthesis():
    gr.Markdown("## **Input Settings**")
    with gr.Row():
        with gr.Column():
            source_prompt = gr.Textbox(
                label="Source Prompt",
                value='a photo of green ducks walking on street',
                interactive=True)
            target_prompt = gr.Textbox(
                label="Target Prompt",
                value='a photo of rubber ducks walking on street',
                interactive=True)
            with gr.Row():
                ddim_steps = gr.Slider(label="DDIM Steps",
                                        minimum=1,
                                        maximum=999,
                                        value=50,
                                        step=1)
                replace_scale = gr.Slider(label="Attention map Replacing Scale ratio of Editing",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.4,
                                        step=0.1)
                replace_layer = gr.Slider(label="Layers to Edit",
                                        minimum=0,
                                        maximum=64,
                                        value=32,
                                        step=8)
            run_btn = gr.Button()
        with gr.Column():
            negative_prompt = gr.Textbox(label="Negative Prompt", value='')
            with gr.Row():
                image_resolution = gr.Slider(label="Image Resolution",
                                            minimum=256,
                                            maximum=768,
                                            value=512,
                                            step=64)
                scale = gr.Slider(label="CFG Scale",
                                minimum=0.1,
                                maximum=30.0,
                                value=7.5,
                                step=0.1)
                seed = gr.Slider(label="Seed",
                                minimum=-1,
                                maximum=2147483647,
                                value=42,
                                step=1)

    gr.Markdown("## **Output**")
    with gr.Row():
        image_source = gr.Image(label="Source Image")
        image_results = gr.Image(label="Image with Editing")

    inputs = [
        source_prompt, target_prompt, replace_scale, replace_layer,
        image_resolution, ddim_steps, scale, seed,negative_prompt
    ]
    run_btn.click(consistent_synthesis, inputs,
                    [image_source, image_results])

    gr.Examples(
        [[
            "a photo of green ducks walking on street",
            "a photo of rubber ducks walking on street",
            42
        ],
            [
                "a photo of a husky",
                "a photo of a poodle", 42
            ],
            [
                "a photo of a white horse",
                "a photo of a zebra in the grass", 42
            ]],
        [source_prompt, target_prompt, seed],
    )


def load_image(image_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)

def resize_array(input_array, target_shape):
    new_array = ndimage.zoom(input_array, (target_shape[0]/input_array.shape[0], target_shape[1]/input_array.shape[1], 1), order=1)
    return new_array

def real_image_editing(source_image, target_prompt,
                       replace_scale, replace_layers, ddim_steps, scale, seed,
                    negative_prompt):

    seed_everything(seed)
    h,w,c = source_image.shape
    target_shape = (h, w)  
    with torch.no_grad():
        ref_prompt = ""
        prompts = [ref_prompt, target_prompt]

        # invert the image into noise map
        if isinstance(source_image, np.ndarray):
            source_image = torch.from_numpy(source_image).to(device) / 127.5 - 1.
            source_image = source_image.unsqueeze(0).permute(0, 3, 1, 2)
            source_image = F.interpolate(source_image, (512, 512))

        start_code, latents_list = model.invert(source_image,
                                                ref_prompt,
                                                guidance_scale=scale,
                                                num_inference_steps=ddim_steps,
                                                return_intermediates=True)
        
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        self_replace_steps = replace_scale
        NUM_DIFFUSION_STEPS = ddim_steps
        controller = SelfAttentionControlEdit(prompts, NUM_DIFFUSION_STEPS,replace_layers, self_replace_steps=self_replace_steps)
        register_attention_control_new(model, controller)

        # inference the synthesized image
        if not negative_prompt == '':
            image_results = model(prompts,
                                latents=start_code,
                                guidance_scale=scale,
                                ref_intermediate_latents=latents_list,
                                neg_prompt = negative_prompt)
        else:
            image_results = model(prompts,
                                latents=start_code,
                                guidance_scale=scale,
                                ref_intermediate_latents=latents_list
                                )
        image_results = image_results.cpu().permute(0, 2, 3, 1).numpy()
        

        image_results_resize_org = resize_array(image_results[0], target_shape)
        image_results_resize_dst = resize_array(image_results[1], target_shape)
    return [
        image_results_resize_org,
        image_results_resize_dst
    ] 


def create_demo_editing():

    gr.Markdown("## **Input Settings**")
    with gr.Row():
        with gr.Column():
            source_image = gr.Image(label="Source Image", value=os.path.join(os.path.dirname(__file__), "examples/img/face.jpg"), interactive=True)
            target_prompt = gr.Textbox(label="Target Prompt",
                                    value='10 years old girl',
                                    interactive=True)
            with gr.Row():
                ddim_steps = gr.Slider(label="DDIM Steps",
                                    minimum=1,
                                    maximum=999,
                                    value=50,
                                    step=1)
                replace_scale = gr.Slider(label="Attention map Replacing Scale ratio of Editing",
                                        minimum=0.0,
                                        maximum=1.0,
                                        value=0.8,
                                        step=0.1)
                replace_layer = gr.Slider(label="Layers to Edit",
                                        minimum=0,
                                        maximum=64,
                                        value=32,
                                        step=8)
            run_btn = gr.Button()
        with gr.Column():
            negative_prompt = gr.Textbox(label="Negative Prompt", value='')
            with gr.Row():
                scale = gr.Slider(label="CFG Scale",
                                minimum=0.1,
                                maximum=30.0,
                                value=7.5,
                                step=0.1)
                seed = gr.Slider(label="Seed",
                                minimum=-1,
                                maximum=2147483647,
                                value=42,
                                step=1)

    gr.Markdown("## **Output**")
    with gr.Row():
        image_recons = gr.Image(label="Source Image")
        image_results = gr.Image(label="Image with Editing")

    inputs = [
        source_image, target_prompt, replace_scale, replace_layer, ddim_steps,
        scale, seed, negative_prompt
    ]
    run_btn.click(real_image_editing, inputs,
                [image_recons, image_results])

    gr.Examples(
        [[os.path.join(os.path.dirname(__file__), "examples/img/face.jpg"),
            "10 year old girl"],
        [os.path.join(os.path.dirname(__file__), "examples/img/girl.png"),
            "smiling woman"],
        ],
        [source_image, target_prompt]
    )


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Tab("Synthesis_image_editing"):
            create_demo_synthesis()
        with gr.Tab("Real_image_editing"):
            create_demo_editing()
    return ui

if __name__ == "__main__":
    args = parser.parse_args()
    model_path = args.model_path
    scheduler = DDIMScheduler(beta_start=0.00085,
                            beta_end=0.012,
                            beta_schedule="scaled_linear",
                            clip_sample=False,
                            set_alpha_to_one=False)
    model = FreePromptPipeline.from_pretrained(model_path,
                                            scheduler=scheduler).to(device)
    demo_editing = add_tab()
    demo_editing.launch()


