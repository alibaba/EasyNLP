import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from easynlp.modelzoo import BertConfig, BertModel, BertTokenizer
from easynlp.modelzoo import TransformerConfig, TransformerTokenizer
from easynlp.modelzoo.models.transformer.modeling_transformer import TransformerModel

# pretrained_model_name_or_path = "/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased"

# kwargs = {'architectures': ['BertForMaskedLM'], 'model_type': 'bert', 'transformers_version': '4.6.0.dev0'}
# config = BertConfig(**kwargs)
# print(config)
# model = BertModel.from_pretrained(pretrained_model_name_or_path, config=config)
# print(model)



# vocab_file = '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased/vocab.txt'
# kwargs = {
#     'special_tokens_map_file': None,
#     'tokenizer_file': '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased/tokenizer.json',
#     'name_or_path': '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased'
# }

# tokenizer = BertTokenizer(
#     vocab_file=vocab_file,
#     do_lower_case=True,
#     do_basic_tokenize=True,
#     never_split=None,
#     unk_token='[UNK]',
#     sep_token='[SEP]',
#     pad_token='[PAD]',
#     cls_token='[CLS]',
#     mask_token='[MASK]',
#     tokenize_chinese_chars=True,
#     strip_accents=None,
#     **kwargs)

# text_a = "The Boston Archdiocese has faced waves of scandal that have not only angered victims' advocates but parishioners and some priests."
# text_b = "The waves of scandal angered not only victims' advocates but parishioners and some priests, to the point that Law could no longer run the archdiocese."
# max_seq_length = 128

# encoding = tokenizer(
#     text_a,
#     text_b,
#     padding='max_length',
#     truncation=True,
#     max_length=max_seq_length)

# print(encoding)



vocab_file = '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/transformer/vocab.txt'
codecs_file = '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/transformer/dict.codecs'

tokenizer = TransformerTokenizer(
    vocab_file=vocab_file,
    codecs_file=codecs_file,
    do_lower_case=True,
    do_basic_tokenize=True,
    null_token='__null__',
    bos_token='__start__',
    eos_token='__end__',
    unk_token='__unk__',
    max_ngram_size=-1,
    max_tokens=-1,
    tokenizer='bpe',
    separator='@@')

text = "your persona: my car broke down last week.\nyour persona: i love sports , but rugby is my favorite.\nRugby football\nI have watched it on ESPN2.  Do you know anything about the history of the sport?\napparently there are separate sports that use different rules now\nI wonder why they did that?  Do they play in Winter?  That's my favorite season"
label = "I'm not sure about that by I can tell you that the sport has been around since1968 and was developed by high school students"
max_seq_length = 512
max_label_length = 128

xs_ids = tokenizer(
    text,
    padding=False,
    truncation=True,
    max_length=max_seq_length
)
ys_ids = tokenizer(
    label,
    padding=False,
    truncation=True,
    max_length=max_label_length
)
xs_ids = [xs_ids['input_ids'] for _ in range(8)]
ys_ids = [ys_ids['input_ids'] for _ in range(8)]

print(xs_ids)

xs = torch.LongTensor(xs_ids)
ys = torch.LongTensor(ys_ids)

print(xs)


# pretrained_model_name_or_path = "/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/transformer"
pretrained_model_name_or_path = "/apsarapangu/disk3/xianyu.lzy/miniconda3/envs/parlai/lib/python3.9/site-packages/data/models/tutorial_transformer_generator/model"

kwargs = {'architectures': ['BertForMaskedLM'], 'model_type': 'bert', 'transformers_version': '4.6.0.dev0'}
config = TransformerConfig()
# print(config)
model = TransformerModel.from_pretrained(pretrained_model_name_or_path, config=config)
# print(model)

scores, preds, encoder_states = model(xs, ys=ys)
print(scores)
print(preds)
print(encoder_states)