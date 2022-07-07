# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch Wukong model."""


# -*- coding: utf-8 -*-

# system
from typing import Union, List, Tuple
from collections import OrderedDict
import os
import json
import logging
from logging import Filter
from logging.handlers import QueueHandler, QueueListener
from torch.multiprocessing import Queue
from dataclasses import dataclass
import argparse
import yaml
import pickle
import time
# 处理base64数据时需要
from PIL import Image
import base64
from io import BytesIO

# nn
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

from .configuration_wukong import WukongConfig

# 函数

# non_local
def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp

# non_local
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster



def load(model, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", clip_path=None, bert_path=None):
    """Load CLIP and BERT model weights
    """
    bert_state_dict = torch.load(bert_path, map_location="cpu") if bert_path else None
    clip_state_dict = torch.load(clip_path, map_location="cpu") if clip_path else None

    restore_model(model, clip_state_dict, bert_state_dict).to(device)

    if str(device) == "cpu":
        model.float()
    return model

def restore_model(model, clip_state_dict: dict, bert_state_dict: dict):
    merged_state_dict = {}

    # use clip_state_dict to initialize the image encoder & logit scale
    if clip_state_dict is not None:
        for k, v in clip_state_dict.items():
            if k.startswith("visual") or k == "logit_scale":
                merged_state_dict[k] = v

    # use bert_state_dict to initialize the text encoder
    if bert_state_dict is not None:
        for k, v in bert_state_dict.items():
            if k.startswith("bert"):
                merged_state_dict[k] = v

    convert_weights(model)
    model.load_state_dict(merged_state_dict, strict=False)
    return model.eval()

def get_dataset(args, is_train, max_txt_length=32):
    input_filename = args.train_data if is_train else args.val_data
    img_filename = args.train_img if is_train else args.val_img
    dataset = JsonlDataset(
        input_filename,
        img_filename,
        split="train" if is_train else "val",
        max_txt_length=max_txt_length)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
    

def get_data(args, max_txt_length=32):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, is_train=True, max_txt_length=max_txt_length)
    if args.val_data:
        data["val"] = get_dataset(
            args, is_train=False, max_txt_length=max_txt_length)

    return data

def get_eval_txt_dataset(args, max_txt_length=32,alt_tk=None):
    input_filename = args.text_data
    dataset = EvalTxtDataset(
        input_filename,
        max_txt_length=max_txt_length,alt_tk=alt_tk)
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.text_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_eval_img_dataset(args):
    img_filename = args.image_data
    dataset = EvalImgDataset(
        img_filename)
    num_samples = len(dataset)
    print('num_samples ',num_samples)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.img_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def setup_primary_logging(log_file, level):
    log_queue = Queue(-1)

    file_handler = logging.FileHandler(filename=log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', 
        datefmt='%Y-%m-%d,%H:%M:%S')

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    file_handler.setLevel(level)
    stream_handler.setLevel(level)

    listener = QueueListener(log_queue, file_handler, stream_handler)

    listener.start()

    return log_queue

def setup_worker_logging(rank, log_queue, level):
    queue_handler = QueueHandler(log_queue)

    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)

    queue_handler.setLevel(level)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)

    root_logger.setLevel(level)

# 类

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model,eps=1e-07)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model,eps=1e-07)
        self.attn_mask = attn_mask
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisualTransformer(nn.Module):
    def __init__(self, 
    input_resolution: int, 
    patch_size: int, 
    width: int, 
    layers: int, 
    heads: int, 
    output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width,eps=1e-07)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width,eps=1e-07)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextTransformer(nn.Module):
    def __init__(self,
                 context_length,
                 vocab_size,
                 output_dim,
                 width,
                 layers,
                 heads,
                 return_full_embed=True):
        super(TextTransformer, self).__init__()
        self.width = width
        self.layers = layers
        self.vocab_size = vocab_size
        self.return_full_embed = return_full_embed

        self.transformer = Transformer(width, layers, heads, self.build_attntion_mask(context_length))
        self.text_projection = torch.nn.Parameter(
            torch.tensor(np.random.normal(0, self.width ** -0.5, size=(self.width, output_dim)).astype(np.float32)))
        self.ln_final = nn.LayerNorm(width,eps=1e-07)

        # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/27
        # https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L22 
        self.embedding_table = nn.Parameter(nn.init.trunc_normal_(torch.empty(vocab_size, width),std=0.02))
        # self.embedding_table = nn.Embedding.from_pretrained(nn.init.trunc_normal_(torch.empty(vocab_size, width),std=0.02))
        self.positional_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(context_length, width),std=0.01))
        # self.positional_embedding = nn.Embedding.from_pretrained(nn.init.trunc_normal_(torch.empty(context_length, width),std=0.01))
        
        self.index_select=torch.index_select
        self.reshape=torch.reshape
    
    @staticmethod
    def build_attntion_mask(context_length):
        mask = np.triu(np.full((context_length, context_length), -np.inf).astype(np.float32), 1)
        mask = torch.tensor(mask)
        return mask

    def forward(self, x: torch.Tensor):

        tail_token=(x==102).nonzero(as_tuple=True)
        bsz, ctx_len = x.shape
        flatten_id = x.flatten()
        index_select_result = self.index_select(self.embedding_table,0, flatten_id)
        x = self.reshape(index_select_result, (bsz, ctx_len, -1))
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x=x[tail_token]
        x = x @ self.text_projection
        return x

class WukongModel(nn.Module):
    def __init__(self,_config,ckpt_path,text_only=False):
        super().__init__()

        self.vision_param={}
        self.text_param={}

        if ckpt_path[-4:]=='.bin':
            with open(ckpt_path, 'rb') as ckpt_fp:
                self.param_dict = torch.load(ckpt_fp,map_location='cpu')
            for key in self.param_dict.keys():
                if 'model.visual_encoder.' in key:
                    new_key=key[len('model.visual_encoder.'):]
                    self.vision_param[new_key]=self.param_dict[key]
                if 'model.text_encoder.' in key:
                    new_key=key[len('model.text_encoder.'):]
                    self.text_param[new_key]=self.param_dict[key]
            self.logit_scale = nn.Parameter(self.param_dict['model.logit_scale'])
        elif ckpt_path[-3:]=='.pt':
            with open(ckpt_path, 'rb') as ckpt_fp:
                self.ckpt = torch.load(ckpt_fp,map_location='cpu')
                self.param_dict=self.ckpt['state_dict']
            for key in self.param_dict.keys():
                if 'module.visual_encoder.' in key:
                    new_key=key[len('module.visual_encoder.'):]
                    self.vision_param[new_key]=self.param_dict[key]
                if 'module.text_encoder.' in key:
                    new_key=key[len('module.text_encoder.'):]
                    self.text_param[new_key]=self.param_dict[key]
            self.logit_scale = nn.Parameter(self.param_dict['module.logit_scale'])
        else:
            with open(ckpt_path, 'rb') as ckpt_fp:
                self.param_dict = pickle.load(ckpt_fp)

            self.name_mapping={'transformer.token_embedding.weight':  'embedding_table',
                    'transformer.positional_embedding':  'positional_embedding',
                    'transformer.text_projection':  'text_projection',
                    'transformer.ln_final.weight':  'ln_final.weight',
                    'transformer.ln_final.bias':  'ln_final.bias',
                    'loss.logit_scale':''}
            for key in self.param_dict.keys():
                if 'visual.' in key:
                    new_key=key[len('visual.'):]
                    self.vision_param[new_key]=torch.tensor(self.param_dict[key])
                else:
                    if key in self.name_mapping:
                        if self.name_mapping[key] !='':
                            self.text_param[self.name_mapping[key]]=torch.tensor(self.param_dict[key])
                    else:
                        self.text_param[key]=torch.tensor(self.param_dict[key])

            self.logit_scale = nn.Parameter(torch.tensor(self.param_dict['loss.logit_scale']))

        if text_only is False:
            self.visual_encoder = VisualTransformer(**_config.data['model']['visual'])
            self.visual_encoder.load_state_dict(self.vision_param)
        
        self.text_encoder=TextTransformer(**_config.data['model']['text'])
        self.text_encoder.load_state_dict(self.text_param)

    def forward(self, image, text):
        assert image is not None and text is not None, "text and image should both not be None!"

        image_features = self.visual_encoder(image)
        text_features = self.text_encoder(text)

        image_features = image_features / image_features.norm(p=2,dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2,dim=-1, keepdim=True)
        
        return image_features, text_features, self.logit_scale.exp()


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler





