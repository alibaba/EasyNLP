import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from easynlp.modelzoo import BertConfig, BertModel

pretrained_model_name_or_path = "/home/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased"

kwargs = {'architectures': ['BertForMaskedLM'], 'model_type': 'bert', 'transformers_version': '4.6.0.dev0'}
config = BertConfig(**kwargs)
print(config)
model = BertModel.from_pretrained(pretrained_model_name_or_path, config=config)
print(model)