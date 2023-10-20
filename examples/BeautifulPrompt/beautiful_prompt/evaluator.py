import os
from typing import Any, List, Mapping, Union

import clip
import ImageReward as RM
import torch
import torch.nn as nn
import torch.nn.functional as F
from ImageReward.ImageReward import MLP as RM_MLP
from PIL import Image
from transformers import AutoModel, AutoProcessor



class Evaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def check_imgs(self, imgs: Union[List[str], List[Image.Image]]):
        new_imgs = []
        for img in imgs:
            if isinstance(img, str):
                assert os.path.isfile(img)
                pil_image = Image.open(img)
                new_imgs.append(pil_image)
            elif isinstance(img, Image.Image):
                new_imgs.append(img)
            else:
                raise TypeError(r'This imgs parameter type has not been supportted yet. Please pass PIL.Image or file path str.')

        return new_imgs

    def forward(self, prompts, imgs):
        '''
        Compute scores based on the given prompts and images.

        Args:
            prompts: A list of prompts.
            imgs: A list of images.

        Returns:
            A list of scores for each <prompt, image> pair.
        '''
        raise NotImplementedError()

class ImageReward(Evaluator):
    '''Reference: https://github.com/THUDM/ImageReward.'''

    def __init__(self, checkpoint: str='ImageReward-v1.0', device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        super().__init__()
        self.model = RM.load(checkpoint, device=device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, prompts, imgs):
        '''
        Reference: https://github.com/THUDM/ImageReward/blob/main/ImageReward/ImageReward.py#L84.
        '''

        assert isinstance(prompts, list)
        assert isinstance(imgs, list)
        assert len(prompts) == len(imgs)
        imgs = self.check_imgs(imgs)

        # text encode
        text_input = self.model.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(self.device)
        
        imgs = torch.stack([self.model.preprocess(img) for img in imgs]).to(self.model.device)
        img_embeds = self.model.blip.visual_encoder(imgs)
        
        # text encode cross attention with image
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(self.device)
        
        text_output = self.model.blip.text_encoder(text_input.input_ids,
                                                attention_mask=text_input.attention_mask,
                                                encoder_hidden_states=img_embeds,
                                                encoder_attention_mask=img_atts,
                                                return_dict=True)
        
        txt_features = text_output.last_hidden_state[:, 0, :].float() # (feature_dim)
        scores = self.model.mlp(txt_features).squeeze(dim=1)
        scores = (scores - self.model.mean) / self.model.std

        return scores.cpu().tolist()

class CLIPScore(Evaluator):
    def __init__(self, device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        super().__init__()
        self.model = RM.load_score('CLIP', device=device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, prompts, imgs):
        '''
        Reference: https://github.com/THUDM/ImageReward/blob/main/ImageReward/models/CLIPScore.py.
        '''

        assert isinstance(prompts, list)
        assert isinstance(imgs, list)
        assert len(prompts) == len(imgs)
        
        texts = clip.tokenize(prompts, truncate=True).to(self.device)
        
        txt_features = F.normalize(self.model.clip_model.encode_text(texts)).float()
        
        imgs = self.check_imgs(imgs)
        
        imgs = torch.stack([self.model.preprocess(img) for img in imgs]).to(self.device)
        img_features = F.normalize(self.model.clip_model.encode_image(imgs)).float()
        
        rewards = torch.sum(torch.mul(txt_features, img_features), dim=1, keepdim=True)
        rewards = torch.squeeze(rewards, dim=1)
        return rewards.cpu().tolist()
    
class AestheticScore(Evaluator):
    '''Reference: https://github.com/christophschuhmann/improved-aesthetic-predictor.'''

    def __init__(self, device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        super().__init__()
        self.model = RM.load_score('Aesthetic', device=device)
        self.device = device
    
    @torch.no_grad()
    def forward(self, prompts, imgs):
        '''
        Reference: https://github.com/THUDM/ImageReward/blob/main/ImageReward/models/AestheticScore.py#L45.
        '''
        
        assert isinstance(imgs, list)
        
        imgs = self.check_imgs(imgs)

        imgs = torch.stack([self.model.preprocess(img) for img in imgs]).to(self.device)
        img_features = F.normalize(self.model.clip_model.encode_image(imgs)).float()
        
        scores = self.model.mlp(img_features)
        scores = torch.squeeze(scores, dim=-1)

        return scores.cpu().tolist()
    

class PickScore(Evaluator):
    '''Reference: https://github.com/yuvalkirstain/PickScore.'''

    def __init__(self,
                 processor_checkpoint: str = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                 model_checkpoint: str = 'yuvalkirstain/PickScore_v1',
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(processor_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint).eval().to(device)
        self.device = device
        
    @torch.no_grad()
    def forward(self, prompts, imgs):
        '''Reference: https://github.com/yuvalkirstain/PickScore#inference-with-pickscore.'''

        assert isinstance(imgs, list)
        
        imgs = self.check_imgs(imgs)
        
        imgs_inputs = self.processor(
            images=imgs,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)
        
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt',
        ).to(self.device)
        
        img_embs = self.model.get_image_features(**imgs_inputs)
        img_embs = img_embs / torch.norm(img_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = self.model.logit_scale.exp() * (text_embs @ img_embs.T)
        scores = scores.diagonal()

        return scores.cpu().tolist()

class HPS(Evaluator):
    '''Reference: https://github.com/tgxs002/align_sd.'''

    def __init__(self,
                 model_checkpoint: str,
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:

        super().__init__()
        self.model, self.processor = clip.load('ViT-L/14', device=device)
        params = torch.load(model_checkpoint)['state_dict']
        self.model.load_state_dict(params)
        self.model.eval()
        self.device = device
        
    @torch.no_grad()
    def forward(self, prompts, imgs):

        assert isinstance(imgs, list)
    
        imgs = self.check_imgs(imgs)
        
        img_inputs = torch.stack([self.processor(img) for img in imgs]).to(self.device)

        text_inputs = clip.tokenize(prompts).to(self.device)

        image_features = self.model.encode_image(img_inputs)
        text_features = self.model.encode_text(text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        hps = image_features @ text_features.T
        hps = hps.diagonal()

        return hps.cpu().tolist()

class HPSv2(Evaluator):
    '''Reference: https://github.com/tgxs002/HPSv2.'''
    

    def __init__(self,
                 model_checkpoint: str,
                 local_dir: str = None,
                 cache_dir: str = None,
                 device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:

        super().__init__()

        from hpsv2.open_clip import create_model_and_transforms, get_tokenizer

        self.model, _, self.processor = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
            cache_dir=cache_dir,
            local_dir=local_dir
        )
        checkpoint = torch.load(model_checkpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.device = device

    @torch.no_grad()
    def forward(self, prompts, imgs):

        assert isinstance(imgs, list)
        
        imgs = self.check_imgs(imgs)
        
        img_inputs = torch.stack([self.processor(img) for img in imgs]).to(self.device)

        text_inputs = self.tokenizer(prompts).to(self.device)

        with torch.cuda.amp.autocast():
            outputs = self.model(img_inputs, text_inputs)
            image_features, text_features = outputs['image_features'], outputs['text_features']

            score = image_features @ text_features.T
            score = score.diagonal()

        return score.cpu().tolist()
