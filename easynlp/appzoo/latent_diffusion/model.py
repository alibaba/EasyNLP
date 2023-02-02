# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import literal_eval
import numpy as np
import torch
import torch.nn as nn
import os
import math
from torch import Tensor
from typing import List, Optional
import json
from tqdm import tqdm, trange
from PIL import Image
from ...utils import losses, get_pretrain_model_path, get_args
from ..application import Application
from collections import OrderedDict 

class Config_Wrapper:
    def __init__(self,json_data):
        self.json_data=json_data

    def to_json_string(self):
        json_str=json.dumps(self.json_data,ensure_ascii=False)
        return json_str

try:

    from ...modelzoo.models.latent_diffusion.ddpm import LatentDiffusionModel
    from ...modelzoo.models.latent_diffusion.autoencoder import AutoencoderKL
    from ...modelzoo.models.latent_diffusion.wukong import FrozenWukongCLIPTextEmbedder
    from ...modelzoo.models.latent_diffusion.plms import PLMSSampler
    from ...modelzoo.models.latent_diffusion.RRDBNet_arch import ESRGAN
    from ...modelzoo.models.latent_diffusion.modules import FrozenCLIPEmbedder

    Key_Class_Mapping={
        'FrozenCLIPEmbedder':FrozenCLIPEmbedder,
        'AutoencoderKL':AutoencoderKL
    }

    class LatentDiffusion(Application):

        @classmethod
        def from_pretrained(self, pretrained_model_name_or_path,args, user_defined_parameters={}):
            instance=LatentDiffusion(pretrained_model_name_or_path,args,user_defined_parameters)
            return instance

        def __init__(self, pretrained_model_name_or_path=None,args=None,user_defined_parameters=None,**kwargs):
            super().__init__()

            if pretrained_model_name_or_path is not None:
                pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
                # 先处理配置，再决定后续如何加载权重
                with open(pretrained_model_name_or_path+'/config.json','r') as config_handle:
                    self.raw_config=json.load(config_handle)
                    # print(self.config)
                checkpoint = torch.load(os.path.join(pretrained_model_name_or_path,'pytorch_model.bin'), map_location=torch.device('cpu'))
                
                if "state_dict" not in checkpoint:
                    sd = checkpoint
                else:
                    sd = checkpoint["state_dict"]
                all_params=self.raw_config["model"]["params"]
            
                self.config=Config_Wrapper(self.raw_config)# used by trainer

                # 权重放在此处统一管理
                # 一阶段权重
                all_params["first_stage_config"]["params"]["ckpt_path"]=None
                all_params["first_stage_model"]=AutoencoderKL(**all_params["first_stage_config"]["params"])
                # 条件阶段权重
                all_params["cond_stage_config"]["params"]["version"]= pretrained_model_name_or_path
                all_params["cond_stage_model"]=FrozenWukongCLIPTextEmbedder(**all_params["cond_stage_config"]["params"])
                
                ## 使用pipeline generator时，修改use_checkpoint=False
                if user_defined_parameters.get('not_use_gradient_checkpoint',False):
                    all_params["unet_config"]["params"]["use_checkpoint"]= False

                self.model=LatentDiffusionModel(**all_params)

                del self.config.json_data["model"]["params"]["first_stage_model"]
                del self.config.json_data["model"]["params"]["cond_stage_model"]

                m, u = self.model.load_state_dict(sd, strict=False)
                if len(m) > 0:
                    print("missing keys:")
                    print(m)
                if len(u) > 0:
                    print("unexpected keys:")
                    print(u)
                self.model.eval()
                _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                self.model = self.model.to(_device)
                self.sr_model = ESRGAN(os.path.join(pretrained_model_name_or_path,'RRDB_ESRGAN_x4.pth'), _device)
                self.sampler = PLMSSampler(self.model)
                self.scale=float(user_defined_parameters.pop('scale',5.0))
                self.n_samples=int(user_defined_parameters.pop('n_samples',4))
                self.n_iter=int(user_defined_parameters.pop('n_iter',1))
                self.H=int(user_defined_parameters.pop('H',256))
                self.W=int(user_defined_parameters.pop('W',256))
                self.ddim_steps=int(user_defined_parameters.pop('ddim_steps',20))
                self.ddim_eta=float(user_defined_parameters.pop('ddim_eta',0.0))
                self.image_prefix=user_defined_parameters.pop('image_prefix','./')
                self.do_sr=user_defined_parameters.pop('do_sr',False)
                if 'write_image' in user_defined_parameters:
                    self.write_image=True
                else:
                    self.write_image=False
                    
        def reset_params(self,n_samples,sample_steps):
            self.n_samples = n_samples
            self.ddim_steps = sample_steps
            print("Reset image sampling number and steps params Done.")

        def forward(self, inputs):
            x, c = self.model.get_input(inputs, self.model.first_stage_key)
            t = torch.randint(0, self.model.num_timesteps, (x.shape[0],), device=self.model.device).long()
            if self.model.cond_stage_trainable:
                c = self.model.get_learned_conditioning(c)
            if self.model.shorten_cond_schedule:  # TODO: drop this option
                tc = self.model.cond_ids[t].to(self.device)
                c = self.model.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

            noise = torch.randn_like(x)
            x_noisy = self.model.q_sample(x_start=x, t=t, noise=noise)
            logits = self.model.apply_model(x_noisy, t, c)
            target = noise

            return logits, target

        @torch.no_grad()    
        def forward_predict(self, inputs):
            all_samples=list()
            for one_input in inputs:
                with torch.no_grad():
                    with self.model.ema_scope():
                        uc = None
                        if self.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.n_samples * [""])
                        for n in trange(self.n_iter, desc="Sampling"):
                            c = self.model.get_learned_conditioning(self.n_samples * [one_input["text"]])
                            shape = [4, self.H//8, self.W//8]
                            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.ddim_eta)
                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                            if self.do_sr is True:
                                x_samples_ddim = self.sr_model.super_resolution(x_samples_ddim)
                            all_samples.append({'idx':one_input["idx"],'image_tensor':x_samples_ddim,'text':one_input["text"]})
            return all_samples

        def compute_loss(self, logits, target, mean=True):
            if self.model.loss_type == 'l1':
                loss = (target - logits).abs()
                if mean:
                    loss = loss.mean()
            elif self.model.loss_type == 'l2':
                if mean:
                    loss = torch.nn.functional.mse_loss(target, logits)
                else:
                    loss = torch.nn.functional.mse_loss(target, logits, reduction='none')
            else:
                raise NotImplementedError("unknown loss type '{loss_type}'")
            return {'loss': loss}

    class StableDiffusion(Application):

        @classmethod
        def from_pretrained(self, pretrained_model_name_or_path,args, user_defined_parameters={}):
            instance=StableDiffusion(pretrained_model_name_or_path,args,user_defined_parameters)
            return instance

        def __init__(self, pretrained_model_name_or_path=None,args=None,user_defined_parameters=None):
            super().__init__()

            if pretrained_model_name_or_path is not None:
                pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
                # 先处理配置，再决定后续如何加载权重
                with open(pretrained_model_name_or_path+'/config.json','r') as config_handle:
                    self.config=json.load(config_handle)
                    all_params=self.config["model"]["params"]
                    first_stage_class=Key_Class_Mapping[all_params["first_stage_config"]["target"]]
                    all_params["first_stage_model"]=first_stage_class(**all_params["first_stage_config"]["params"])
                    cond_stage_class=Key_Class_Mapping[all_params["cond_stage_config"]["target"]]
                    all_params["cond_stage_config"]["params"]={"version":os.path.join(pretrained_model_name_or_path,'clip-vit-large-patch14')}
                    all_params["cond_stage_model"]=cond_stage_class(**all_params["cond_stage_config"]["params"])
                
                ## 使用pipeline generator时，修改use_checkpoint=False
                if user_defined_parameters.get('not_use_gradient_checkpoint',False):
                    all_params["unet_config"]["params"]["use_checkpoint"]= False
                    
                checkpoint = torch.load(os.path.join(pretrained_model_name_or_path,'pytorch_model.bin'), map_location=torch.device('cpu'))
                if "state_dict" not in checkpoint:
                    sd = checkpoint
                else:
                    sd = checkpoint["state_dict"]
                self.model=LatentDiffusionModel(**all_params)
                m, u = self.model.load_state_dict(sd, strict=False)
                # print('----------------------------------------------')
                if len(m) > 0:
                    print("missing keys:")
                    print(m)
                if len(u) > 0:
                    print("unexpected keys:")
                    print(u)
                self.model.eval()
                _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                self.model = self.model.to(_device)
                self.sr_model = ESRGAN(os.path.join(pretrained_model_name_or_path,'RRDB_ESRGAN_x4.pth'), _device)
                self.sampler = PLMSSampler(self.model)
                self.scale=float(user_defined_parameters.pop('scale',5.0))
                self.n_samples=int(user_defined_parameters.pop('n_samples',4))
                self.n_iter=int(user_defined_parameters.pop('n_iter',1))
                self.H=int(user_defined_parameters.pop('H',512))
                self.W=int(user_defined_parameters.pop('W',512))
                self.ddim_steps=int(user_defined_parameters.pop('ddim_steps',100))
                self.ddim_eta=float(user_defined_parameters.pop('ddim_eta',0.0))
                self.image_prefix=user_defined_parameters.pop('image_prefix','./')
                self.do_sr=user_defined_parameters.pop('do_sr',False)
                if 'write_image' in user_defined_parameters:
                    self.write_image=True
                else:
                    self.write_image=False
        def reset_params(self,n_samples,sample_steps):
            self.n_samples = n_samples
            self.ddim_steps = sample_steps
            print("Reset image sampling number and steps params Done.")
            
        def forward_predict(self, inputs):
            all_samples=list()
            for one_input in inputs:
                with torch.no_grad():
                    with self.model.ema_scope():
                        uc = None
                        if self.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.n_samples * [""])
                        for n in trange(self.n_iter, desc="Sampling"):
                            c = self.model.get_learned_conditioning(self.n_samples * [one_input["text"]])
                            shape = [4, self.H//8, self.W//8]
                            samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.ddim_eta)
                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                            if self.do_sr is True:
                                x_samples_ddim = self.sr_model.super_resolution(x_samples_ddim)
                            all_samples.append({'idx':one_input["idx"],'image_tensor':x_samples_ddim,'text':one_input["text"]})
            return all_samples

        def compute_loss(self, forward_outputs, label_ids, **kwargs):
            pass

except Exception as ex:

    class LatentDiffusion(Application):
        @classmethod
        def from_pretrained(self, pretrained_model_name_or_path,args, user_defined_parameters={}):
            instance=LatentDiffusion(pretrained_model_name_or_path,args,user_defined_parameters)
            return instance
        def __init__(self, pretrained_model_name_or_path=None,args=None,user_defined_parameters=None):
            super().__init__()


    class StableDiffusion(Application):
        @classmethod
        def from_pretrained(self, pretrained_model_name_or_path,args, user_defined_parameters={}):
            instance=StableDiffusion(pretrained_model_name_or_path,args,user_defined_parameters)
            return instance
        def __init__(self, pretrained_model_name_or_path=None,args=None,user_defined_parameters=None):
            super().__init__()

    print("Latent Diffusion Models are not supported. If you do not use these models, please ignore this message. Exception: %s" % ex)


