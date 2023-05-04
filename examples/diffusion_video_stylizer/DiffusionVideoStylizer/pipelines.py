import torch, PIL, cv2
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from typing import List, Union
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import VaeImageProcessor

from annotator.midas import MidasDetector
from annotator.hed import HEDdetector
from annotator.util import resize_image

from diffusers.models.transformer_2d import Transformer2DModel
from DiffusionVideoStylizer.models import Transformer2DModel_edited


def convert_Transformer2DModel(model: Transformer2DModel, torch_dtype=torch.float32):
    
    # Extract config and state dict from UNet
    config = model.config
    state_dict = model.state_dict()
    
    # Replace the Transformer2DModel
    model_ = Transformer2DModel_edited.from_config(config)
    model_.load_state_dict(state_dict)
    model_ = model_.to(torch_dtype)
    
    return model_


def convert_UNet2DConditionModel(unet: UNet2DConditionModel, torch_dtype=torch.float32):
    
    # down
    for i in range(len(unet.down_blocks)):
        if hasattr(unet.down_blocks[i], "attentions"):
            for j in range(len(unet.down_blocks[i].attentions)):
                if isinstance(unet.down_blocks[i].attentions[j], Transformer2DModel):
                    unet.down_blocks[i].attentions[j] = convert_Transformer2DModel(unet.down_blocks[i].attentions[j], torch_dtype=torch_dtype)

    # mid
    if hasattr(unet.mid_block, "attentions"):
        for j in range(len(unet.mid_block.attentions)):
            if isinstance(unet.mid_block.attentions[j], Transformer2DModel):
                unet.mid_block.attentions[j] = convert_Transformer2DModel(unet.mid_block.attentions[j], torch_dtype=torch_dtype)

    # up
    for i in range(len(unet.up_blocks)):
        if hasattr(unet.up_blocks[i], "attentions"):
            for j in range(len(unet.up_blocks[i].attentions)):
                if isinstance(unet.up_blocks[i].attentions[j], Transformer2DModel):
                    unet.up_blocks[i].attentions[j] = convert_Transformer2DModel(unet.up_blocks[i].attentions[j], torch_dtype=torch_dtype)
                    
    return unet


def convert_ControlNetModel(controlnet: ControlNetModel, torch_dtype=torch.float32):
    # down
    for i in range(len(controlnet.down_blocks)):
        if hasattr(controlnet.down_blocks[i], "attentions"):
            for j in range(len(controlnet.down_blocks[i].attentions)):
                if isinstance(controlnet.down_blocks[i].attentions[j], Transformer2DModel):
                    controlnet.down_blocks[i].attentions[j] = convert_Transformer2DModel(controlnet.down_blocks[i].attentions[j], torch_dtype=torch_dtype)

    # mid
    if hasattr(controlnet.mid_block, "attentions"):
        for j in range(len(controlnet.mid_block.attentions)):
            if isinstance(controlnet.mid_block.attentions[j], Transformer2DModel):
                controlnet.mid_block.attentions[j] = convert_Transformer2DModel(controlnet.mid_block.attentions[j], torch_dtype=torch_dtype)
    
    return controlnet


class ControlnetImageProcesserDepth:
    def __init__(self, resolution=512):
        self.apply_midas = MidasDetector()
        self.resolution = resolution

    def __call__(self, image):
        size  = image.size
        image = np.array(image)
        image = resize_image(image, self.resolution)
        image, _ = self.apply_midas(image)
        image = Image.fromarray(image)
        image = image.resize(size)
        return image

    
class ControlnetImageProcesserHED:
    def __init__(self, resolution=512):
        self.apply_hed = HEDdetector()
        self.resolution = resolution

    def __call__(self, image):
        size  = image.size
        image = np.array(image)
        image = resize_image(image, self.resolution)
        image = self.apply_hed(image)
        image = Image.fromarray(image)
        image = image.resize(size)
        return image


class VideoStylizingPipeline(DiffusionPipeline):
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: List[ControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
    ):
        # Convert model to support cross-frame attention
        unet = convert_UNet2DConditionModel(unet, torch_dtype=unet.dtype)
        controlnet = [convert_ControlNetModel(model, torch_dtype=model.dtype) for model in controlnet]
        controlnet = MultiControlNetModel(controlnet)
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.controlnet_image_processor = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(controlnet.dtype),
        ])
        
        
    def get_image_height_width(
        self,
        image: PIL.Image.Image
    ):
        height, width = image.height//8*8, image.width//8*8
        return height, width
    
    
    def get_text_embedding(
        self,
        prompt: str,
    ):
        length = self.tokenizer.model_max_length
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        max_length = (input_ids.shape[1] + length - 1) // length * length
        
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True).input_ids
        input_ids = rearrange(input_ids, "B (N L) -> (B N) L", L=length)
        prompt_embed = self.text_encoder(input_ids.to(self.device))[0]
        prompt_embed = rearrange(prompt_embed, "(B N) L D -> B (N L) D", B=1)
        
        return prompt_embed
    
    
    def prepare_image(
        self,
        frames: List[PIL.Image.Image],
        processer = lambda image: image,
        height = None,
        width = None,
    ):
        images = []
        for image_ in tqdm(frames):
            image_ = processer(image_)
            image_ = image_.convert("RGB")
            image_ = image_.resize((width, height), resample=PIL.Image.LANCZOS)
            image_ = self.controlnet_image_processor(image_)
            images.append(image_)
            
        images = torch.stack(images)
        return images
    
    
    def get_latent_image(
        self,
        image: PIL.Image.Image,
        height: int = None,
        width: int = None,
        generator = None
    ):
        image = image.resize((width, height), resample=PIL.Image.LANCZOS)
        image = self.vae_image_processor.preprocess(image)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        
        latent_image = self.vae.encode(image).latent_dist.sample(generator)
        latent_image = self.vae.config.scaling_factor * latent_image
        latent_image = latent_image.to("cpu")
        
        return latent_image
    
    
    def get_noise(
        self,
        shape,
        dtype = torch.float32
    ):
        noise = torch.randn(shape, dtype=dtype)
        noise = noise * self.scheduler.init_noise_sigma
        return noise
    
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    
    def combine_image(
        self,
        pattern,
        image_id,
        images,
        image_reference
    ):
        image_combined = []
        for p in pattern:
            if type(p) is str and p=="reference":
                image_combined.append(image_reference)
            elif type(p) is int:
                if image_id+p<0 or image_id+p>=images.shape[0]:
                    image_combined.append(image_reference)
                else:
                    image_combined.append(images[image_id+p: image_id+p+1])
        image_combined = torch.cat(image_combined)
        return image_combined
    
    
    def get_timesteps_for_img2img(
        self,
        num_inference_steps,
        img2img_strength,
    ):
        deoise_steps = int(num_inference_steps * img2img_strength)

        t_start = max(num_inference_steps - deoise_steps, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps
    

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = None,
        negative_prompt: str = "",
        frames: List[PIL.Image.Image] = None,
        image_reference: PIL.Image.Image = None,
        controlnet_scale: Union[float, List[float]] = 1.0,
        image_processers = None,
        num_inference_steps: int = 50,
        seed_of_noise = None,
        guidance_scale: float = 7.5,
        combine_pattern = ["reference", -1, 0, 1],
        img2img_strength = 1.0,
    ):
        # parameters
        height, width = self.get_image_height_width(frames[0])
        device = self.device
        if isinstance(controlnet_scale, float):
            controlnet_scale = [controlnet_scale] * len(self.controlnet.nets)
            
        # Encode input prompt
        prompt_embed_posi = self.get_text_embedding(prompt)
        prompt_embed_nega = self.get_text_embedding(negative_prompt)
        
        # Prepare controlnet input
        controlnet_frames = [self.prepare_image(frames, p, height, width) for p in image_processers]
        
        # Prepare reference image
        latent_image_reference = self.get_latent_image(image_reference, height, width)
        controlnet_frame_reference = [self.prepare_image([image_reference], p, height, width) for p in image_processers]
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.get_timesteps_for_img2img(num_inference_steps, img2img_strength)
        
        # Prepare latent variables
        if seed_of_noise is not None:
            torch.manual_seed(seed_of_noise)
        noise = self.get_noise(
            shape=(1, self.unet.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
            dtype=self.unet.dtype
        )
        if img2img_strength<1.0:
            latent_images = torch.concat([self.get_latent_image(image, height, width) for image in tqdm(frames)])
            latents = self.scheduler.add_noise(latent_images, torch.cat([noise] * len(frames)), timesteps[0])
        else:
            latents = torch.cat([noise] * len(frames))
        
        # Denoising loop
        progress_bar = tqdm(total=len(timesteps)*len(frames))
        for t_id, t in enumerate(timesteps):
            noise_pred_list = []
            for image_id in range(len(frames)):
                # Input
                unet_input = self.combine_image(
                    combine_pattern,
                    image_id,
                    latents,
                    self.scheduler.add_noise(latent_image_reference, noise, t)
                )
                controlnet_input = []
                for controlnet_id in range(len(self.controlnet.nets)):
                    controlnet_input.append(self.combine_image(
                        combine_pattern,
                        image_id,
                        controlnet_frames[controlnet_id],
                        controlnet_frame_reference[controlnet_id]
                    ))
                unet_input = unet_input.to(device)
                controlnet_input = [i.to(device) for i in controlnet_input]
                    
                # ControlNet
                down_res_posi, mid_res_posi = self.controlnet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embed_posi,
                    controlnet_cond=controlnet_input,
                    conditioning_scale=controlnet_scale,
                    return_dict=False,
                )
                down_res_nega, mid_res_nega = self.controlnet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embed_nega,
                    controlnet_cond=controlnet_input,
                    conditioning_scale=controlnet_scale,
                    return_dict=False,
                )
                
                # UNet
                noise_pred_posi = self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embed_posi,
                    down_block_additional_residuals=down_res_posi,
                    mid_block_additional_residual=mid_res_posi,
                ).sample
                noise_pred_nega = self.unet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embed_nega,
                    down_block_additional_residuals=down_res_nega,
                    mid_block_additional_residual=mid_res_nega,
                ).sample
                
                # perform guidance
                noise_pred = noise_pred_nega + guidance_scale * (noise_pred_posi - noise_pred_nega)
                noise_pred = noise_pred[combine_pattern.index(0)]
                noise_pred_list.append(noise_pred.to("cpu"))
                
                progress_bar.update(1)
                
            noise_pred = torch.stack(noise_pred_list)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        images = []
        for image_id in tqdm(range(len(frames))):
            images.append(self.decode_latents(latents[image_id: image_id+1].to(device)))
        images = np.concatenate(images)
        
        images = self.numpy_to_pil(images)
        
        return dict(images=images)
