import torch
from easynlp.modelzoo import TransformerConfig, TransformerTokenizer, TransformerModel


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