from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.schedulers.scheduling_ddim import ConfigMixin, register_to_config, BaseOutput, KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class DDIMSchedulerOutput(BaseOutput):

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class SkipableDDIMScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = 1.0
        self.init_noise_sigma = 1.0

        # setable values
        self.set_timesteps(10)

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        return sample

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        self.timesteps = torch.arange(self.config.num_train_timesteps-1, -1, -step_ratio)

    def denoise(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        alpha_prod_t,
        alpha_prod_t_prev
    ):
        weight_e = (1 - alpha_prod_t_prev) ** (0.5) - (alpha_prod_t_prev * (1 - alpha_prod_t) / alpha_prod_t) ** (0.5)
        weight_x = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)
        
        prev_sample = sample * weight_x + model_output * weight_e
        
        weight_e = - ((1 - alpha_prod_t) / alpha_prod_t) ** (0.5)
        weight_x = (1 / alpha_prod_t) ** (0.5)

        pred_original_sample = sample * weight_x + model_output * weight_e

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        if isinstance(timestep, torch.Tensor) and len(timestep.shape)>0:
            timestep = timestep[0]
        alpha_prod_t = self.alphas_cumprod[timestep]
        timestep_prev = timestep - self.config.num_train_timesteps // self.num_inference_steps
        if timestep_prev < 0:
            alpha_prod_t_prev = self.final_alpha_cumprod
        else:
            alpha_prod_t_prev = self.alphas_cumprod[timestep_prev]

        return self.denoise(model_output, timestep, sample, alpha_prod_t, alpha_prod_t_prev)


    def return_to_timestep(
        self,
        timestep: int,
        sample: torch.FloatTensor,
        sample_stablized: torch.FloatTensor,
    ):
        alpha_prod_t = self.alphas_cumprod[timestep]
        noise_pred = (sample - alpha_prod_t ** (0.5) * sample_stablized) / ((1 - alpha_prod_t) ** (0.5))
        return noise_pred


    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps

