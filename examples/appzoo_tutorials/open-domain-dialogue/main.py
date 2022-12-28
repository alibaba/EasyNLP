import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from easynlp.modelzoo import BertConfig, BertModel
from easynlp.modelzoo import BertTokenizer, TransformerTokenizer

# pretrained_model_name_or_path = "/home/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased"

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
max_seq_length = 512

encoding = tokenizer(
    text,
    padding=False,
    truncation=True,
    max_length=max_seq_length
)

print(encoding)