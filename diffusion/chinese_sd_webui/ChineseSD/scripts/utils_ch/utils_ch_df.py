# coding:utf-8
import gradio as gr
import os,tarfile
import torch
import requests
import json
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline,StableDiffusionImg2ImgPipeline,StableDiffusionControlNetPipeline,ControlNetModel,StableDiffusionInpaintPipelineLegacy
from diffusers import EulerAncestralDiscreteScheduler,HeunDiscreteScheduler,DDIMScheduler,DDPMScheduler,EulerDiscreteScheduler,DPMSolverMultistepScheduler,PNDMScheduler
from modules.shared import opts, cmd_opts
from modules.ui_components import ToolButton, DropdownMulti
from transformers import DPTImageProcessor, DPTForDepthEstimation
from modules import devices, lowvram, script_callbacks, shared, paths,scripts
from modules.api import api
import cv2
import base64

##master dir of chinsese diffusion.
# Contains chinese base diffusion ,chinese controlnet models,chinese lora models and so on
if shared.cmd_opts.chinese_diffusion_dir_master is not None:
    models_dir_master = shared.cmd_opts.chinese_diffusion_dir_master 
elif hasattr(shared.cmd_opts, 'public_cache') and shared.cmd_opts.public_cache:
    models_dir_master = '/stable-diffusion-cache/models/ChineseDiffusion'
else:
    models_dir_master = os.path.join(paths.models_path, "ChineseDiffusion")


##chinsese diffusion models,
# contains pai-diffusion-artist-large-zh and pai-diffusion-artist-xlarge-zh
model_dir_diffusion = "Chinese_diffusion"
if shared.cmd_opts.chinese_diffusion_dir is not None:
    stable_diffusion_dir = shared.cmd_opts.chinese_diffusion_dir 
else:
    stable_diffusion_dir = os.path.join(models_dir_master, model_dir_diffusion)

##chinsese controlnet models
# contains canny and depth and a img2depth generation model(dpt-large)
model_dir_controlnet = "Chinese_Controlnet"
if shared.cmd_opts.chinese_controlnet_dir is not None:
    controlnet_model_dir = shared.cmd_opts.chinese_controlnet_dir 
else:
    controlnet_model_dir = os.path.join(models_dir_master, model_dir_controlnet)
##chinsese lora model
# Contains only chinese poem lora model
model_dir_lora = "Chinese_Lora"
if shared.cmd_opts.chinese_lora_dir is not None:
    lora_model_dir = shared.cmd_opts.chinese_lora_dir 
else:
    lora_model_dir = os.path.join(models_dir_master, model_dir_lora)

available_models = []
available_controlnet = []
available_lora = []


def model_list(models_dir):
    available_models = []
    if not os.path.exists(models_dir):
            os.makedirs(models_dir,exist_ok=True)
    for dirname in os.listdir(models_dir):
        if dirname =='dpt-large':
                continue
        if os.path.isdir(os.path.join(models_dir, dirname)):
            available_models.append(dirname)
    
    return available_models

def list_available_models():
    global available_models
    available_models.clear()
    available_models.append('None')

    if not os.path.exists(stable_diffusion_dir):
        os.makedirs(stable_diffusion_dir,exist_ok=True)
    for dirname in os.listdir(stable_diffusion_dir):
        if os.path.isdir(os.path.join(stable_diffusion_dir, dirname)):
            available_models.append(dirname)
    

def list_available_controlnet():
    global available_controlnet
    available_controlnet.clear()
    available_controlnet.append('None')
       
    if not os.path.exists(controlnet_model_dir):
        os.makedirs(controlnet_model_dir,exist_ok=True)
    for dirname in os.listdir(controlnet_model_dir):
        if dirname =='dpt-large':
            continue
        if os.path.isdir(os.path.join(controlnet_model_dir, dirname)): 
            available_controlnet.append(dirname)
    
def list_lora_models():
    global available_lora
    available_lora.clear()
    available_lora.append('None')
    
    if not os.path.exists(lora_model_dir):
        os.makedirs(lora_model_dir,exist_ok=True)
    for dirname in os.listdir(lora_model_dir):
        if os.path.isdir(os.path.join(lora_model_dir, dirname)):
            available_lora.append(dirname)


def select_pipe(model_name):
    model_dir = os.path.join(stable_diffusion_dir, model_name)
    if os.path.exists(model_dir):
        pipeline = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
    
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'
    elif model_name =='pai-diffusion-artist-large-zh':
        pipeline = StableDiffusionPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh', torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
    
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name =='pai-diffusion-artist-xlarge-zh':
        pipeline = StableDiffusionPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-xlarge-zh',torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
    
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'
    else:
        return None,'Can not finding the model: '+ model_name + ' in path of: '+ model_dir + '. Please check it and download your model.'

def select_img2img_pipe(model_name):

    model_dir = os.path.join(stable_diffusion_dir, model_name)

    if os.path.exists(model_dir):
        print('load model now')

        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")

        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name =='pai-diffusion-artist-large-zh':
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh', torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")

        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    elif model_name =='pai-diffusion-artist-xlarge-zh':
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-xlarge-zh')
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")

        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    else:
        return None,'Can not finding the model: '+ model_name + ' in path of: '+ model_dir + '. Please check it and download your model.'



def select_inpainting_pipe(model_name):
    model_dir = os.path.join(stable_diffusion_dir, model_name)

    if os.path.exists(model_dir):
        pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained(model_dir, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
    
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'
    elif model_name =='pai-diffusion-artist-large-zh':
        pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh', torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
    
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'
    elif model_name =='pai-diffusion-artist-xlarge-zh':
        pipeline = StableDiffusionInpaintPipelineLegacy.from_pretrained('alibaba-pai/pai-diffusion-artist-xlarge-zh')
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
    
        return pipeline, 'Loading model : ' + model_name + ' done. Enjoy it!'

    else:
        return None,'Can not finding the model: '+ model_name + ' in path of: '+ model_dir + '. Please check it and download your model.'

def select_controlnet_pipe(model_name,control_model):
    model_dir = os.path.join(stable_diffusion_dir, model_name)

    controlnet, warning_inf = select_controlnet(control_model)
    if os.path.exists(model_dir):
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(model_dir, controlnet=controlnet, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to("cuda")
        return pipeline,'Loading model : ' + model_name + ' done. '+ str(warning_inf)+ '. Enjoy it!'
    else:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained('alibaba-pai/pai-diffusion-artist-large-zh', controlnet=controlnet, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        pipeline = pipeline.to("cuda")
        return pipeline,'Loading model : ' + model_name + ' done. '+ str(warning_inf)+ '. Enjoy it!'


def select_controlnet(control_model):
    control_model_dir = os.path.join(controlnet_model_dir, control_model)
    if os.path.exists(control_model_dir):
        controlnet = ControlNetModel.from_pretrained(control_model_dir,torch_dtype=torch.float16)
        return controlnet, 'The current control model: ' + control_model
    else:
        return None, 'controlnet model: '+ control_model + ' is not exist now, please download it first!'


def refresh_model(model_name,tag,control_model,lora,whether_lora):
    if not model_name:
        return 'Select the model you need.',gr.Slider.update()
    if not(control_model =='txt2img' or control_model =='img2img' or control_model =='inpainting'):
        if not control_model:
            return 'The current selected base chinses model is '+ str(model_name)+'. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                                                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.' ,gr.Slider.update()
        if control_model=='None':
            return 'The current selected base chinses model is '+ str(model_name)+'. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                                                              ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.' ,gr.Slider.update()

    if model_name =='pai-diffusion-anime-large-zh':
        return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(value=768)
    else:
        return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(value=512)
        
def refresh_model_controlnet(model_name,tag,control_model,lora,whether_lora):
    if not model_name:
        return 'Select the model you need.',gr.Slider.update(),gr.CheckboxGroup.update()
    if model_name=='None':
        return 'The current control_model is' + str(control_model) + '. Please select the model you need.',gr.Slider.update(),gr.CheckboxGroup.update()

    if model_name =='pai-diffusion-anime-large-zh':
        if not control_model:
            return 'The current selected base chinses model is '+ str(model_name)+'. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                                                            ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.' ,gr.Slider.update(),gr.CheckboxGroup.update()
        elif control_model=='None':
            return 'The current selected base chinses model is '+ str(model_name)+'. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                                                            ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.' ,gr.Slider.update(),gr.CheckboxGroup.update()
        elif control_model =='pai-diffusion-artist-large-zh-controlnet-canny':
            return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(value=768),gr.CheckboxGroup.update(value='canny')
        elif control_model =='pai-diffusion-artist-large-zh-controlnet-depth':
            return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(value=768),gr.CheckboxGroup.update(value='depth')
        else:
            return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(),gr.CheckboxGroup.update()
    else:
        if not control_model:
            return 'The current selected base chinses model is '+ str(model_name)+'. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                                                            ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.' ,gr.Slider.update(),gr.CheckboxGroup.update()
        elif control_model=='None':
            return 'The current selected base chinses model is '+ str(model_name)+'. Please choice the controlnet model and the control mode before using it. Please note that pai-diffusion-artist-large-zh-controlnet-canny and pai-diffusion-artist-large-zh-controlnet-depth ' \
                                                            ' are based on the pai-diffusion-artist-large-zh, choose other based models may cause unexpected outputs.' ,gr.Slider.update(),gr.CheckboxGroup.update()
        elif control_model =='pai-diffusion-artist-large-zh-controlnet-canny':
            return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(),gr.CheckboxGroup.update(value='canny')
        elif control_model =='pai-diffusion-artist-large-zh-controlnet-depth':
            return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(),gr.CheckboxGroup.update(value='depth')
        else:
            return refresh_checkpoints(model_name,tag,control_model,lora,whether_lora),gr.Slider.update(),gr.CheckboxGroup.update()

    
def refresh_lora_model(model_name,tag,control_model,lora,whether_lora):

    return refresh_checkpoints_lora(model_name,tag,control_model,lora,whether_lora)

def refresh_checkpoints(model_name,tag,control_model,lora,whether_lora):
    global pipe
    # print(tag)
    if tag =='txt2img': 
        pipe,warning_inf = select_pipe(model_name)

        if not lora=='None' and lora:
            lora_path = os.path.join(lora_model_dir, lora)

            pipe.unet.load_attn_procs(lora_path)
            if not model_name =='pai-diffusion-artist-large-zh':
                warning_inf_lora = 'The current lora model: '+ str(lora) + '. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d ' \
                                                              ' are based on the pai-diffusion-artist-large-zh, choose other models may cause unexpected outputs.'
            else:
                warning_inf_lora = 'The current lora model: '+ str(lora) + '.'
            warning_inf = warning_inf + ' ' + warning_inf_lora
        return warning_inf

    if tag =='img2img': 
        pipe,warning_inf = select_img2img_pipe(model_name)
        if not lora=='None' and lora:
            lora_path = os.path.join(lora_model_dir, lora)

            pipe.unet.load_attn_procs(lora_path)
            if not model_name =='pai-diffusion-artist-large-zh':
                warning_inf_lora = 'The current lora model: '+ str(lora) + '. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d ' \
                                                              ' are based on the pai-diffusion-artist-large-zh, choose other models may cause unexpected outputs.'
            else:
                warning_inf_lora = 'The current lora model: '+ str(lora) + '.'
            warning_inf = warning_inf + ' ' + warning_inf_lora
        return warning_inf

    if tag =='inpainting': 
        pipe,warning_inf = select_inpainting_pipe(model_name)
        return warning_inf

    if tag =='controlnet': 
        pipe,warning_inf = select_controlnet_pipe(model_name,control_model)
        return warning_inf
        
def refresh_checkpoints_lora(model_name,tag,control_model,lora,whether_lora):
    global pipe

    if not model_name:#in case of only select lora, no based model is selected.
        
        warning_inf = 'The current lora model: ' + str(lora) +'. Now, select the chinsese diffusion model you need. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d are based on the pai-diffusion-artist-large-zh,' \
                                                              ' choose other models may cause an unexpected error.'
    # print(tag)
    elif model_name=='None':
        return 'The current lora model: ' + str(lora) +'. Now, select the chinsede diffusion model you need. Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d are based on the pai-diffusion-artist-large-zh,' \
                                                              ' choose other models may cause an unexpected error.'
    elif lora=='None' and model_name: #in case of from using lora to free lora
        warning_inf = refresh_checkpoints(model_name, tag, control_model, lora,whether_lora)
    else:#in case of select model and further to use lora
        lora_path = os.path.join(lora_model_dir, lora)

        if tag =='txt2img': 
            model_dir = os.path.join(stable_diffusion_dir, model_name)
            if os.path.exists(model_dir):
                pipe.unet.load_attn_procs(lora_path)
                if model_name =='pai-diffusion-artist-large-zh':
                    warning_inf =  'The current lora model: '+ str(lora) + '. Enjoy it!'
                else:
                    warning_inf =  'The current lora model: '+ str(lora) + '. Enjoy it! Note that pai-diffusion-artist-large-zh-lora-poem and pai-diffusion-artist-large-zh-lora-25d are based on the pai-diffusion-artist-large-zh,' \
                                                              ' choose other models may cause an unexpected error.'
            else:
                warning_inf = 'Can not finding the model: '+ model_name + ' in path of: '+ model_dir + '. Please check it and download your model.'

        if tag =='img2img': 
            model_dir = os.path.join(stable_diffusion_dir, model_name)
            if os.path.exists(model_dir):
                pipe.unet.load_attn_procs(lora_path)
                warning_inf = 'The current lora model: '+ str(lora) + '. Enjoy it!'
            else:
                warning_inf = 'Can not finding the model: '+ model_name + ' in path of: '+ model_dir + '. Please check it and download your model.'
    return warning_inf

def to_canny(image):
    low_threshold = 100
    high_threshold = 200

    if type(image) is Image.Image:
        image = np.array(image)
    if type(image) is np.ndarray:
        print('image type after: {}'.format(type(image)))
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image
    else:
        return ''
    
def to_depth(image):
    processor = DPTImageProcessor.from_pretrained(os.path.join(controlnet_model_dir,'dpt-large'))
    model = DPTForDepthEstimation.from_pretrained(os.path.join(controlnet_model_dir,'dpt-large'))
    image = Image.fromarray(image)
    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    image= Image.fromarray(formatted)

    return image


def infer_text2img(model_name, prompt, negative_prompt, height,width,guide, steps,num_images,seed,scheduler,use_Lora):
    # seed_everything(seed)
    if not negative_prompt:
        negative_prompt = ''
    # pipe = select_pipe(model_name)
    scheduler = scheduler
    if scheduler =='DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Euler a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)       
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    torch.manual_seed(seed)
    sample_images = num_images
    
    #Forward embeddings and negative embeddings through text encoder
    if len(prompt)>=33:
        max_length = pipe.tokenizer.model_max_length
        
        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        # print('input_ids: {}'.format(input_ids))
        if len(prompt) < len(negative_prompt):
            negative_prompt = negative_prompt[0:len(prompt)]
        negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
        negative_ids = negative_ids.to("cuda")

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
        if use_Lora:
            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,height=height,width=width,guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps,cross_attention_kwargs={"scale": 0.4}).images
        else:
            image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,height=height,width=width,guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps).images
    else:
        if use_Lora:
            image = pipe(prompt=prompt,negative_prompt=negative_prompt,height=height,width=width,guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps,cross_attention_kwargs={"scale": 0.4}).images
        else:
            image = pipe(prompt=prompt,negative_prompt=negative_prompt,height=height,width=width,guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps).images

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])
        
    return image_out

def infer_inpainting(model_name, prompt,negative_prompt, image_in,mask_in, height,width,strength,num_images,guide, steps,scheduler,seed):
    # seed_everything(seed)
    if not negative_prompt:
        negative_prompt = ''
    # pipe = select_pipe(model_name)

    scheduler = scheduler
    if scheduler =='DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Euler a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)       
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    image = Image.fromarray(image_in)
    mask = Image.fromarray(mask_in)

    image_in = image.convert("RGB").resize((width,height))
    mask_in = mask.convert('L').resize((width,height),resample=Image.Resampling.NEAREST)    
    torch.manual_seed(seed)
    sample_images = num_images

    image = pipe(prompt=prompt, image=image_in,mask_image=mask_in,strength=strength, guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps).images

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])        
    return image_out

def infer_controlnet(control_mode, prompt, negative_prompt,image_in,height,width,guide, steps,num_images,seed,scheduler):
    # seed_everything(seed)
    if not negative_prompt:
        negative_prompt = ''
    # pipe = select_pipe(model_name)
    scheduler = scheduler
    if scheduler =='DPM':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Euler a':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='Heun':
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='DDIM':
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='DDPM':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif scheduler =='PNDM':
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)       
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print(f'control mode: {control_mode[0]}')
    if control_mode[0] =='canny':
        image_in = to_canny(image_in)
    elif control_mode[0] =='depth':
        image_in = to_depth(image_in)
    else:
        image_in = to_canny(image_in)
    print('control mode trans Done')

    torch.manual_seed(seed)
    sample_images = num_images
    
    #Forward embeddings and negative embeddings through text encoder
    if len(prompt)>=33:
        max_length = pipe.tokenizer.model_max_length
        
        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        # print('input_ids: {}'.format(input_ids))
        if len(prompt) < len(negative_prompt):
            negative_prompt = negative_prompt[0:len(prompt)]
        negative_ids = pipe.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=input_ids.shape[-1], return_tensors="pt").input_ids                                                                                                     
        negative_ids = negative_ids.to("cuda")

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
        image = pipe(prompt_embeds=prompt_embeds,image= image_in,negative_prompt_embeds=negative_prompt_embeds,height=height,width=width,guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps).images
    else:
        print('before inference')
        image = pipe(prompt=prompt,image= image_in,negative_prompt=negative_prompt,height=height,width=width,guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps).images
    print('after inference')

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])
        
    return image_out
    
def infer_img2img(model_name,prompt,image_in,height,width,num_images,guide,steps,strength,seed,use_Lora):


    image = Image.fromarray(image_in)
    image_in = image.convert("RGB")
    w, h = map(lambda x: x - x % 32, (height, width))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image_in = torch.from_numpy(image)
    image_in = 2.*image_in - 1.

    torch.manual_seed(seed)
    sample_images = num_images

    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained("model_name").to("cuda")
    if use_Lora:
        image = pipe(prompt=prompt, image=image_in, strength=strength, guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps,cross_attention_kwargs={"scale": 0.4}).images
    else:
        image = pipe(prompt=prompt, image=image_in, strength=strength, guidance_scale=guide,num_images_per_prompt=sample_images,num_inference_steps=steps).images

    image_out = []
    for k in range(sample_images):
        image_out.append(image[k])        
    return image_out




