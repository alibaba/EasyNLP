from ..application import Application
from ...utils import losses, get_pretrain_model_path, get_args
from ...modelzoo.models.wukong.modeling_wukong import WukongModel
from ...modelzoo.models.wukong.configuration_wukong import WukongConfig
import torch
import json

class WukongCLIP(Application):
    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path, user_defined_parameters={},**kwargs):
        instance=WukongCLIP(pretrained_model_name_or_path,user_defined_parameters)
        return instance

    def __init__(self, pretrained_model_name_or_path=None,user_defined_parameters=None, **kwargs):
        super().__init__()
        pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
        if pretrained_model_name_or_path is not None:
            # 临时代码，用于将pickle wukong 转成标准模型
            # 勿删，可能接其他悟空权重时会用到
            # import yaml
            # config_path=pretrained_model_name_or_path+'/wukong_vit_l_14_clip.yaml'
            # with open(config_path, 'r') as stream:
            #     _config = yaml.safe_load(stream)
            #     _config['model']['visual'].pop('type', None)
            #     _config['model']['visual'].pop('return_full_embed', None)
            #     _config['model']['visual'].pop('token_learner', None)
            #     if "heads" not in _config['model']['visual']:
            #         _config['model']['visual']["heads"] = _config['model']['visual']["width"] // 64
            #     _config['model']['text'].pop('type', None)
            # # print(self.config)
            # self.config=WukongConfig(_config)
            # self.model=WukongModel(self.config,pretrained_model_name_or_path+'/wukong_vit_l_14_clip.pkl')
            # for name,param in self.model.named_parameters():
            #     param.requires_grad = False 
            
            config_path=pretrained_model_name_or_path+'/config.json'
            with open(config_path, 'r') as stream:
                _config = json.load(stream)
                self.config=WukongConfig(_config)
            self.model=WukongModel(self.config,pretrained_model_name_or_path+'/pytorch_model.bin')
            self.loss_img = torch.nn.CrossEntropyLoss()
            self.loss_txt = torch.nn.CrossEntropyLoss()

    def forward(self, inputs):
        if 'pixel_values' in inputs:
            image_features = self.model.visual_encoder(inputs['pixel_values'].to(self.model.logit_scale.device))
            image_features = image_features / image_features.norm(p=2,dim=-1, keepdim=True)
        else:
            image_features=None
        
        if 'input_ids' in inputs:
            text_features = self.model.text_encoder(inputs['input_ids'].to(self.model.logit_scale.device))
            text_features = text_features / text_features.norm(p=2,dim=-1, keepdim=True)
        else:
            text_features=None 
        
        return {'image_features':image_features, 'text_features':text_features,'logit_scale': self.model.logit_scale.exp()},[]

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        image_features=forward_outputs['image_features']
        text_features=forward_outputs['text_features']
        logit_scale=forward_outputs['logit_scale'].mean()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.arange(len(logits_per_image)).long().to(image_features.device)
        total_loss = (
            self.loss_img(logits_per_image, ground_truth)
            + self.loss_txt(logits_per_text, ground_truth)
            ) / 2

        return {'loss': total_loss}
