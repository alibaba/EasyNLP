import os
import torch
import torch.nn.functional as F

from ..application import Application
from ...modelzoo import AutoConfig
from .vqgan import VQModel
from ...modelzoo.models.artist.modeling_artist import GPT
from ...modelzoo.models.artist.modeling_artist_knowl import GPT_knowl
from ...modelzoo.models.artist.configuration_artist import ARTISTConfig
from ...utils.logger import logger



class TextImageGeneration(Application):
    def __init__(self, pretrained_model_name_or_path=None, user_defined_parameters=None, **kwargs):
        super().__init__()
        self.first_stage_key = "image"
        self.cond_stage_key = "text"
        
        if pretrained_model_name_or_path is not None: 
            self.first_stage_model = VQModel()
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.transformer = GPT(self.config)
            self.init_from_ckpt(pretrained_model_name_or_path)
        else:
            logger.info("Train Text2Image model from scratch....")
            vqgan_ckpt_path = user_defined_parameters.get('vqgan_ckpt_path')
            self.first_stage_model = VQModel(ckpt_path=vqgan_ckpt_path).eval()
            img_vocab_size = int(user_defined_parameters.get('img_vocab_size', '16384'))
            text_vocab_size = int(user_defined_parameters.get('text_vocab_size', '21128'))
            vocab_size = img_vocab_size + text_vocab_size
            block_size = int(user_defined_parameters.get('text_len', '32')) + int(user_defined_parameters.get('img_len', '256'))
            n_layer = int(user_defined_parameters.get('n_layer', '12'))
            n_head = int(user_defined_parameters.get('n_head', '12'))
            n_embd = int(user_defined_parameters.get('n_embd', '768'))
            self.config = ARTISTConfig(vocab_size=vocab_size, img_vocab_size=img_vocab_size, text_vocab_size=text_vocab_size, \
                                        block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
            self.transformer = GPT(self.config)
        self.pkeep = kwargs.get('pkeep', 1.0)


    def init_from_ckpt(self, pretrained_model_name_or_path, **kwargs):
        weight_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        sd = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {pretrained_model_name_or_path}")
        return self

    
    def forward(self, inputs):
        x = inputs['image']
        c = inputs['text']
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)       # z_indice: torch.Size([batch_size, 256]) 
        c_indices = c

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        # print("cz_indices = ", cz_indices.shape)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]
        return logits, target
    
    def generate(self, inputs, top_k=100, temperature=1.0):
        # print("inputs = ", inputs.shape)
        cidx = inputs

        sample = True
        steps = 256

        for k in range(steps):
            x_cond = cidx
            logits, _ = self.transformer(x_cond)
            # print("logits shape = ", logits.shape)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            cidx = torch.cat((cidx, ix), dim=1)
        img_idx = cidx[:, 32:]
        # print("cidx shape   = ", img_idx.shape)

        return img_idx

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        # indices = self.permuter(indices)
        return quant_z, indices
        

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        # index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        # quant_c, c_indices = self.encode_to_c(c)
        c_indices = c

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x
    
    def get_text(self, key, batch):
        x = batch[key]
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_text(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def compute_loss(self, logits, target, **kwargs):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return {'loss': loss}

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

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
    


class TextImageGeneration_knowl(Application):
    def __init__(self, pretrained_model_name_or_path=None, user_defined_parameters=None, **kwargs):
        super().__init__()
        self.first_stage_key = "image"
        self.cond_stage_key = "text"
        
        if pretrained_model_name_or_path is not None: 
            self.first_stage_model = VQModel()
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            self.transformer = GPT_knowl(self.config)
            self.init_from_ckpt(pretrained_model_name_or_path)
        else:
            logger.info("Train Text2Image model from scratch....")
            vqgan_ckpt_path = user_defined_parameters.get('vqgan_ckpt_path')
            self.first_stage_model = VQModel(ckpt_path=vqgan_ckpt_path).eval()
            img_vocab_size = int(user_defined_parameters.get('img_vocab_size', '16384'))
            text_vocab_size = int(user_defined_parameters.get('text_vocab_size', '21128'))
            vocab_size = img_vocab_size + text_vocab_size
            block_size = int(user_defined_parameters.get('text_len', '32')) + int(user_defined_parameters.get('img_len', '256'))
            n_layer = int(user_defined_parameters.get('n_layer', '12'))
            n_head = int(user_defined_parameters.get('n_head', '12'))
            n_embd = int(user_defined_parameters.get('n_embd', '768'))
            self.config = ARTISTConfig(vocab_size=vocab_size, img_vocab_size=img_vocab_size, text_vocab_size=text_vocab_size, \
                                        block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
            self.transformer = GPT_knowl(self.config)
        self.pkeep = kwargs.get('pkeep', 1.0)


    def init_from_ckpt(self, pretrained_model_name_or_path, **kwargs):
        weight_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        sd = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {pretrained_model_name_or_path}")
        return self

    
    def forward(self, inputs):
        x = inputs['image']
        c = inputs['text']
        words_emb = inputs['words_emb']

        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)       # z_indice: torch.Size([batch_size, 256]) 
        c_indices = c

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        # print("cz_indices = ", cz_indices.shape)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1], words_emb, flag=True)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]
        return logits, target
    
    def generate(self, cidx, words_emb, top_k=100, temperature=1.0):
        # print("inputs = ", inputs.shape)

        sample = True
        steps = 256

        for k in range(steps):
            x_cond = cidx
            logits, _ = self.transformer(x_cond, words_emb, flag=bool(k==0))
            # print("logits shape = ", logits.shape)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            cidx = torch.cat((cidx, ix), dim=1)
        img_idx = cidx[:, 32:]
        # print("cidx shape   = ", img_idx.shape)

        return img_idx

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            # one pass suffices since input is pure noise anyway
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            #noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            # take all logits for now and scale by temp
            logits = logits / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # cut off conditioning
            x = ix[:, c.shape[1]-1:]
        else:
            for k in range(steps):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
                logits, _ = self.transformer(x_cond)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            # cut off conditioning
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        # indices = self.permuter(indices)
        return quant_z, indices
        

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        # index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = 4
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        # quant_c, c_indices = self.encode_to_c(c)
        c_indices = c

        # create a "half"" sample
        z_start_indices = z_indices[:,:z_indices.shape[1]//2]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1]-z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)

        # det sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["samples_half"] = x_sample
        log["samples_nopix"] = x_sample_nopix
        log["samples_det"] = x_sample_det
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x
    
    def get_text(self, key, batch):
        x = batch[key]
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_text(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def compute_loss(self, logits, target, **kwargs):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return {'loss': loss}

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

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
    

    