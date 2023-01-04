from easynlp.modelzoo import TransformerTokenizer

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