import os
import torch
import torch.nn.functional as F

from ..application import Application
from ...modelzoo import AutoConfig
from ...modelzoo import AutoTokenizer
from ...modelzoo.models.mingpt_i2t.modeling_mingpt_i2t import MinGPT
from ...modelzoo.models.mingpt_i2t.configuration_mingpt_i2t import MinGPTI2TConfig
from .clip import load as CLIPFromPretrained
from .tokenizer import ImageTextBERTTokenizer
from ...utils import get_pretrain_model_path
from ...utils.logger import logger


class CLIPGPTFrameTextGeneration(Application):
    def __init__(self, pretrained_model_name_or_path=None, user_defined_parameters=None, **kwargs):
        super().__init__()
        self.cond_stage_key = 'image'
        self.generate_stage_key = 'text'
        self.prefix_encoder_type = 'vit'
        self.pkeep = user_defined_parameters.get('pkeep', 1.0)
        self.device = user_defined_parameters.get('device', 'cuda')

        # when pretraining, the pretrained_model_name_or_path is actually the tokenizer
        # if we don't set pretrained_model_name_or_path, \
        # the vocab.txt will not be automatically copied to the new director of pretrained model
        # Therefore, the pretrained_model_name_or_path should be set in pretraining
        self.enable_pretraining = False
        if pretrained_model_name_or_path is None:
            self.enable_pretraining = True
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if config.model_type in ['bert', 'gpt2']:
                self.enable_pretraining = True

        if self.enable_pretraining:
            # if image_encoder is not parametered, default image encoder is 'vit'
            is_clip_encoder = user_defined_parameters.get('app_parameters', True).get('enable_vit', True)
            assert is_clip_encoder == True, 'wrong image encode settings'

        # CLIP & GPT
        if self.enable_pretraining:
            logger.info("Train Frame2Text model from scratch....")

            # text_tokenizer
            text_tokenizer_path = pretrained_model_name_or_path if pretrained_model_name_or_path else 'bert-base-chinese'
            text_tokenizer_path = get_pretrain_model_path(text_tokenizer_path)
            self.text_tokenizer = ImageTextBERTTokenizer(text_tokenizer_path, start_id = 0)
            text_vocab_size = len(self.text_tokenizer)
  
            # vision_encoder: vit
            vit_ckpt_path = user_defined_parameters.get('vit_ckpt_path')
            # openai_clip: float16 if GPU available, float32 if CPU only. so load the clip model on cpu and then turn to gpu
            self.first_stage_model = CLIPFromPretrained(vit_ckpt_path, jit=False, device=torch.device("cpu"))[0].visual.eval()

            # gpt config
            #text_vocab_size = int(user_defined_parameters.get('text_vocab_size', '21128'))
            #vocab_size = text_vocab_size

            self.text_len = int(user_defined_parameters.get('text_len', '32'))
            self.img_len = int(user_defined_parameters.get('img_len', '256'))
            block_size = self.text_len + self.img_len
            n_layer = int(user_defined_parameters.get('n_layer', '12'))
            n_head = int(user_defined_parameters.get('n_head', '12'))
            n_embd = int(user_defined_parameters.get('n_embd', '768'))
            
            assert self.prefix_encoder_type in vit_ckpt_path.lower(), 'the prefix encoder type is not consistent with ckpt_path'
            self.config = MinGPTI2TConfig(vocab_size=text_vocab_size, block_size=block_size, \
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd, decode_vocab_size=text_vocab_size, \
                    prefix_encoder_type=self.prefix_encoder_type, prefix_encoder_ckpt_path=vit_ckpt_path)
 
            # setting the projection of first_stage_model's outputs according to `n_embd` of GPT
            if n_embd == self.first_stage_model.width:
                self.first_stage_model.proj = None
           
            # gpt
            self.transformer = MinGPT(self.config)

        else:
            # text_tokenizer
            text_tokenizer_path = pretrained_model_name_or_path
            self.text_tokenizer = ImageTextBERTTokenizer(text_tokenizer_path, start_id = 0)

            # gpt config
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            
            # image encoder
            assert self.config.prefix_encoder_type == self.prefix_encoder_type, \
                'This class is not consistent with pretrained model, Please add the user_defined_parameters \"enable_vit=True\"'
            vit_ckpt_path = self.config.prefix_encoder_ckpt_path
            ## openai_clip: float16 if GPU available, float32 if CPU only. so load the clip model on cpu and then turn to gpu
            self.first_stage_model = CLIPFromPretrained(vit_ckpt_path, jit=False, device=torch.device("cpu"))[0].visual.eval()
            # setting the projection of first_stage_model's outputs according to `n_embd` of GPT
            if self.config.n_embd == self.first_stage_model.width:
                self.first_stage_model.proj = None

            # gpt
            self.transformer = MinGPT(self.config)
            # initialize from pretrained model
            self.init_from_ckpt(pretrained_model_name_or_path)
                       
            # sequence length
            self.text_len = int(user_defined_parameters.get('text_len', '32'))
            self.img_len = int(user_defined_parameters.get('img_len', '256'))
            assert self.transformer.block_size == self.text_len + self.img_len, \
                'The text_len or img_len is wrong, the sum of the two value should be equal to block_size of gpt'

        
        # image resize
        img_size = int(user_defined_parameters.get('img_size', '224'))
        assert img_size == self.first_stage_model.input_resolution, 'img_size is not equal to the input_resolution of vit'
        self.img_size = self.first_stage_model.input_resolution

        # image sequence length
        patch_size = self.first_stage_model.patch_size
        assert self.img_len == ((self.img_size/patch_size) * (self.img_size/patch_size)), \
            'the value of \'img_len\' must be equal to the square of vit.input_resolution/vit.patch_size'
        
        # frame length
        self.frame_num = int(user_defined_parameters.get('frame_num', '3'))
        assert self.frame_num <= self.img_len, 'the frame_num should not larger than img_len'


    def init_from_ckpt(self, pretrained_model_name_or_path, **kwargs):
        weight_path = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
        print (f"weight_path from {weight_path}")
        sd = torch.load(weight_path, map_location='cpu')
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {pretrained_model_name_or_path}")
        return self
    
    @torch.no_grad()
    def encode_images_to_embeddings(self, image_pixels):
        return self.first_stage_model(image_pixels) # torch.Size([8, img_len=256, 1024])

    def forward(self, inputs):
        text_tokens_ids = inputs['text']    # x: text_token_ids  [B, text_len=32]
        image_pixels = inputs['image']   # c: image_pixels    [B, frame_num, 3, img_size=224, img_size=224]

        assert (image_pixels.shape[1] == self.frame_num and image_pixels.shape[2] == 3 and image_pixels.shape[3] == self.img_size and \
            image_pixels.shape[4] == self.img_size), 'invalid image shape'
        
        # encode image to visual features
        # 1) convert the shape of image_pixels to (xxxx, 3, img_size, img_size)
        image_pixels = image_pixels.view(-1, 3, self.img_size, self.img_size)  # [B * frame_num, 3, img_size=224, img_size=224]
        # 2) encoder each frame to a single visual feature
        image_embedding_features = self.encode_images_to_embeddings(image_pixels) # [8 * frame_num, 768 or 1024]
        # 3) recover the shape of these frame features
        n_embed_dim = image_embedding_features.shape[-1]
        image_embedding_features = image_embedding_features.view(-1, self.frame_num, n_embed_dim)  # [8, frame_num, 768 or 1024]

        # text
        token_indices = text_tokens_ids   # z_indices: text_token_ids
        
        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(token_indices.shape,
                                                        device=token_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            random_indices = torch.randint_like(token_indices, self.transformer.config.vocab_size)
            all_indices = mask*token_indices+(1-mask)*random_indices
        else:
            all_indices = token_indices            # a_indices: text_token_ids

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = token_indices
        # make the prediction
        logits, _ = self.transformer(idx=all_indices[:, :-1], embeddings = image_embedding_features)
        logits = logits[:, image_embedding_features.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    
    def generate(self, predix_image_embedding, prefix_token_inputs = None, top_k=100, temperature=1.0):
        assert predix_image_embedding.shape[1] == self.frame_num

        x = prefix_token_inputs   # [B, 256]

        sample = True
        steps = self.text_len    # 32
        
        for k in range(steps):
            x_cond = x

            logits, _ = self.transformer(idx = x_cond, embeddings = predix_image_embedding)
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

            # if eos_token was found in one sentence, set sentence to finished
            all_end_tokens = torch.ones_like(ix).mul(self.text_tokenizer.end_token_id).to(self.device)
            # print ('end_flag=', ix==all_end_tokens)
            if x is not None:
                unfinished_tokens = x.mul((ix != all_end_tokens).long())
                # stop when each sentence is finished, or if we exceed the maximum length
                if unfinished_tokens.max() == 0:
                    break

                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)
            else:
                x = ix

        # cut off conditioning
        assert x.shape[1] <= self.text_len
        token_idx = x
        # print (token_idx.shape)
        
        return token_idx

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
