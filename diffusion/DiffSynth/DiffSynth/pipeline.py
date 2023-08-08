import torch
import numpy as np
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange, repeat
from typing import List, Union, Optional, Dict, Any, Tuple
from tqdm import tqdm
from PIL import Image

from diffusers.pipelines.controlnet.pipeline_controlnet import (
    DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, ControlNetModel, KarrasDiffusionSchedulers,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin

from .attention import set_cross_frame_attention
from .scheduler import SkipableDDIMScheduler


class MultiControlNetModel(ModelMixin):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet

    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.

    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """

    def __init__(self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]):
        super().__init__()
        self.global_pool_conditions = [controlnet.config.global_pool_conditions for controlnet in controlnets]
        self.nets = torch.nn.ModuleList(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
        condition_side: bool = True,
    ) -> Tuple:
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                guess_mode,
                return_dict,
            )
            
            # For 'shuffle' model
            if self.global_pool_conditions[i]:
                if condition_side:
                    down_samples = [torch.mean(x, dim=(2, 3), keepdim=True) for x in down_samples]
                    mid_sample = torch.mean(mid_sample, dim=(2, 3), keepdim=True)
                else:
                    down_samples = [torch.zeros_like(x) for x in down_samples]
                    mid_sample = torch.zeros_like(mid_sample)
                    
            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample


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
        set_cross_frame_attention(unet)
        for module in controlnet:
            set_cross_frame_attention(module)
        controlnet = MultiControlNetModel(controlnet)
        scheduler = SkipableDDIMScheduler.from_config(scheduler.config)
        
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
        image: Image.Image
    ):
        height, width = image.height//8*8, image.width//8*8
        return height, width
    
    
    def get_text_embedding(
        self,
        prompt: str,
        batch_size: int = 1
    ):
        length = self.tokenizer.model_max_length
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        max_length = (input_ids.shape[1] + length - 1) // length * length
        
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True).input_ids
        input_ids = rearrange(input_ids, "B (N L) -> (B N) L", L=length)
        prompt_embed = self.text_encoder(input_ids.to(self.device))[0]
        prompt_embed = rearrange(prompt_embed, "(B N) L D -> B (N L) D", B=1)
        
        prompt_embed = repeat(prompt_embed, "B L D -> (B M) L D", M=batch_size)
        return prompt_embed
    
    
    def prepare_image(
        self,
        frames: List[Image.Image],
        processer = lambda image: image,
        height = None,
        width = None,
    ):
        if processer.__class__.__name__.startswith("Video"):
            frames = processer(frames)
            images = []
            for image_ in frames:
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=Image.LANCZOS)
                image_ = self.controlnet_image_processor(image_)
                images.append(image_)
        else:
            images = []
            for image_ in tqdm(frames, "Preparing images for ControlNet"):
                image_ = processer(image_)
                image_ = image_.convert("RGB")
                image_ = image_.resize((width, height), resample=Image.LANCZOS)
                image_ = self.controlnet_image_processor(image_)
                images.append(image_)
        if len(images)==0:
            return None
        images = torch.stack(images)
        return images
    
    
    def get_latent_image(
        self,
        image: Image.Image,
        height: int = None,
        width: int = None,
        generator = None
    ):
        image = image.resize((width, height), resample=Image.LANCZOS)
        image = self.vae_image_processor.preprocess(image)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        
        latent_image = self.vae.encode(image).latent_dist.mean
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
    
    
    def combine_image(
        self,
        pattern,
        image_id,
        images,
        image_reference
    ):
        image_combined = []
        for p in pattern:
            if type(p) is str:
                flag, reference_id = p.split("_")
                if flag!="reference":
                    raise Warning("pattern is not reference")
                reference_id = int(reference_id)
                image_combined.append(image_reference[reference_id: reference_id+1])
            elif type(p) is int:
                if image_id+p<0:
                    image_combined.append(images[:1])
                elif image_id+p>=images.shape[0]:
                    image_combined.append(images[-1:])
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

    
    def get_images_from_latents(
        self,
        latents
    ):
        images = []
        for image_id in tqdm(range(len(latents)), desc="Decoding frames"):
            latent = 1 / self.vae.config.scaling_factor * latents[image_id: image_id+1].to(self.device)
            image = self.vae.decode(latent.to(self.device)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            image = (image * 255).round().astype("uint8")
            image = Image.fromarray(image)
            images.append(image)

        return images


    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        frames: List[Image.Image] = None,
        frames_reference: Image.Image = None,
        controlnet_frames: List = None,
        controlnet_frames_reference: List = None,
        controlnet_processers = None,
        controlnet_scale: Union[float, List[float]] = 1.0,
        num_inference_steps: int = 50,
        seed_of_noise = None,
        guidance_scale: float = 7.5,
        combine_pattern = [0],
        img2img_strength = 1.0,
        flow_frames = None,
        init_latents = None,
        fixed_noise = True,
        smoother = None,
        ignore_smoother_steps = 0,
        smoother_interval = 1,
    ):
        # parameters
        height, width = self.get_image_height_width(frames[0])
        device = self.device
        if isinstance(controlnet_scale, float):
            controlnet_scale = [controlnet_scale] * len(self.controlnet.nets)
            
        # Encode input prompt
        prompt_embed_posi = self.get_text_embedding(prompt, batch_size=len(combine_pattern))
        prompt_embed_nega = self.get_text_embedding(negative_prompt, batch_size=len(combine_pattern))
        
        # Prepare controlnet input
        controlnet_input_frames = []
        for controlnet_frames_list, p in zip(controlnet_frames, controlnet_processers):
            controlnet_input_frames.append(self.prepare_image(controlnet_frames_list, p, height, width))
        
        # Prepare reference image
        if len(frames_reference)>0:
            latent_frames_reference = torch.concat([self.get_latent_image(frame, height, width) for frame in frames_reference])
        else:
            latent_frames_reference = None
        controlnet_input_frames_reference = []
        for f, p in zip(controlnet_frames_reference, controlnet_processers):
            controlnet_input_frames_reference.append(self.prepare_image(f, p, height, width))

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device="cpu")
        timesteps = self.get_timesteps_for_img2img(num_inference_steps, img2img_strength)
        
        # Prepare noise
        if seed_of_noise is not None:
            torch.manual_seed(seed_of_noise)
        if fixed_noise:
            noise = self.get_noise(
                shape=(1, self.unet.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
                dtype=self.unet.dtype
            )
            noise = torch.concat([noise] * len(frames))
        else:
            noise = self.get_noise(
                shape=(len(frames), self.unet.config.in_channels, height // self.vae_scale_factor, width // self.vae_scale_factor),
                dtype=self.unet.dtype
            )
        
        # Prepare latent variables
        if init_latents is None:
            if img2img_strength<1.0:
                latent_images = torch.concat([self.get_latent_image(image, height, width) for image in tqdm(frames, desc="Preparing initial latent variables")])
                latents = self.scheduler.add_noise(latent_images, noise, timesteps[0])
            else:
                latents = noise.clone()
        else:
            latents = init_latents

        if smoother is not None:
            smoother.prepare(flow_frames)
        
        # Denoising loop
        progress_bar = tqdm(total=len(timesteps)*len(frames), desc="Denoising")
        for t_id, t in enumerate(timesteps):
            noise_pred_list = []
            for image_id in range(len(frames)):
                # Input
                if latent_frames_reference is not None:
                    noised_latent_frames_reference = self.scheduler.add_noise(latent_frames_reference, noise[image_id:image_id+1], t)
                else:
                    noised_latent_frames_reference = None 
                unet_input = self.combine_image(
                    combine_pattern,
                    image_id,
                    latents,
                    noised_latent_frames_reference
                )
                controlnet_input = []
                for controlnet_id in range(len(self.controlnet.nets)):
                    controlnet_input.append(self.combine_image(
                        combine_pattern,
                        image_id,
                        controlnet_input_frames[controlnet_id],
                        controlnet_input_frames_reference[controlnet_id]
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
                    condition_side=True
                )
                down_res_nega, mid_res_nega = self.controlnet(
                    unet_input,
                    t,
                    encoder_hidden_states=prompt_embed_nega,
                    controlnet_cond=controlnet_input,
                    conditioning_scale=controlnet_scale,
                    return_dict=False,
                    condition_side=False
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

            if smoother is not None and t_id<len(timesteps)-ignore_smoother_steps and t_id%smoother_interval==0:
                pred_original_sample = self.scheduler.step(noise_pred, t, latents).pred_original_sample
                if smoother.operating_space == "pixel":
                    estimated_frames = self.get_images_from_latents(pred_original_sample)
                    estimated_frames = smoother.smooth(estimated_frames)
                    pred_original_sample = torch.concat([self.get_latent_image(image, height, width) for image in estimated_frames])
                elif smoother.operating_space == "final latent":
                    pred_original_sample = smoother.smooth(pred_original_sample, prompt_embed_posi)
                else:
                    print("operating_space error")
                noise_pred = self.scheduler.return_to_timestep(t, latents, pred_original_sample)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
        progress_bar.close()
            
        images = self.get_images_from_latents(latents)
        
        return dict(images=images)

