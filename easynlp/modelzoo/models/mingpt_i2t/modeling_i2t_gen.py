import os
import torch
import torch.nn.functional as F
from torch import nn

from .modeling_clip import load as CLIPFromPretrained
from .modeling_clip import VisionTransformer
from .modeling_mingpt import MinGPT
from .modeling_vqgan import VQModel
from .modeling_tokenizer import ImageTextBERTTokenizer

from .configuration_mingpt import MinGPTConfig
from .configuration_clip import VisionTransformerConfig
from .configuration_vqgan import VQModelConfig
from .configuration_i2t_gen import I2TGenModelConfig

from ....utils import get_pretrain_model_path
from ....utils.logger import logger
from ..auto.tokenization_auto import AutoTokenizer


class CLIPGPTI2TGenModel(nn.Module):
    def __init__(self, vision_encoder: VisionTransformer, gpt: MinGPT, text_tokenizer_path: str, \
            pkeep: float = 1.0, device: str = 'cuda', **kwargs):
        super().__init__()

        self.first_stage_model = vision_encoder.eval()
        self.transformer = gpt
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path, do_lower_case=True)

        self.pkeep = pkeep
        self.device = device

    def forward(self, image_pixels, text_tokens_ids):
        # image
        assert (image_pixels.shape[1] == 3 and image_pixels.shape[2] == self.img_size and \
            image_pixels.shape[3] == self.img_size), 'invalid image shape'
        print ("image_pixels.shape=", image_pixels.shape)
        # c.shape = [B, 3, 224, 224]  if ViT is ViT_Large_14
        # print (type(self.first_stage_model))
        image_embedding_features = self.first_stage_model(image_pixels)
        # print (image_embedding_features.shape)   # torch.Size([8, 256, 1024])

        # text
        z_indices = text_tokens_ids   # z_indices: text_token_ids
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                        device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices            # a_indices: text_token_ids

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(idx=a_indices[:, :-1], embeddings = image_embedding_features)
        logits = logits[:, image_embedding_features.shape[1]-1:]

        return logits, target


    def decode_to_text(self, index):
        text_list = []
        index_array = index.tolist()
        for row, index_list in enumerate(index_array):
            if self.text_tokenizer.end_token_id in index_list:
                offset = index_list.index(self.text_tokenizer.end_token_id)
            else:
                offset = len(index_list)
            text = self.text_tokenizer.decode(index_list[:offset])
            text_list.append(text)
        return text_list

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
    

    def compute_loss(self, logits, target, **kwargs):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return {'loss': loss}

    #@classmethod
    #def load_from_pretrained_stage1(cls, vision_encoder, gpt, text_tokenizer_path, \
    #    pkeep = 1.0, device = 'cuda', **kwargs):
    #    return cls(vision_encoder, gpt, text_tokenizer_path, pkeep, device, **kwargs)

    # @classmethod
    # def load_from_config(cls, config):
    #     return cls(config.vision_encoder_config, config.mingpt_config, \
    #         config.text_tokenizer_path, config.pkeep, config.device)
    
class VQGANGPTI2TGenModel(nn.Module):
    def __init__(self, vision_encoder: VQModel, gpt: MinGPT, text_tokenizer_path: str, \
            pkeep: float = 1.0, device: str = 'cuda', **kwargs):
        super().__init__()

        self.first_stage_model = vision_encoder.eval()
        self.transformer = gpt
        self.text_tokenizer = ImageTextBERTTokenizer(text_tokenizer_path, start_id = 0)
        self.image_token_start_id = len(self.text_tokenizer)

        self.pkeep = pkeep
        self.device = device


    def forward(self, inputs):
        x = inputs['text']    # x: text_token_ids  [B, 32]
        c = inputs['image']   # c: image_pixels    [B, img_height_size, img_width_size, 3]
        assert (c.shape[1] == 3 and c.shape[2] == self.img_size and c.shape[3] == self.img_size), 'invalid image shape'
        #print ("c.shape=", c.shape)

        z_indices = x   # z_indices: text_token_ids
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                        device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices            # a_indices: text_token_ids
        
        # c = c.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)  #[B, 3, height, width]
        # print (c.shape)   # c.shape = [B, 3, 256, 256]
        _, c_indices = self.encode_to_c(c)   # c_indices: image_token_ids  c_indices.shape=[B, 3, (256/16)^2, (256/16)^2]
        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_z, _, info = self.first_stage_model.encode(c)
        indices = info[2].view(quant_z.shape[0], -1)
        #indices = self.permuter(indices) + self.image_token_start_id
        indices = indices + self.image_token_start_id
        return quant_z, indices

    def decode_to_text(self, index):
        text_list = []
        index_array = index.tolist()
        for row, index_list in enumerate(index_array):
            if self.text_tokenizer.end_token_id in index_list:
                offset = index_list.index(self.text_tokenizer.end_token_id)
            else:
                offset = len(index_list)
            text = self.text_tokenizer.decode(index_list[:offset])
            text_list.append(text)
        return text_list

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
    

    def compute_loss(self, logits, target, **kwargs):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return {'loss': loss}