from easynlp.modelzoo import BertTokenizer

vocab_file = '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased/vocab.txt'
kwargs = {
    'special_tokens_map_file': None,
    'tokenizer_file': '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased/tokenizer.json',
    'name_or_path': '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased'
}

tokenizer = BertTokenizer(
    vocab_file=vocab_file,
    do_lower_case=True,
    do_basic_tokenize=True,
    never_split=None,
    unk_token='[UNK]',
    sep_token='[SEP]',
    pad_token='[PAD]',
    cls_token='[CLS]',
    mask_token='[MASK]',
    tokenize_chinese_chars=True,
    strip_accents=None,
    **kwargs)

text_a = "The Boston Archdiocese has faced waves of scandal that have not only angered victims' advocates but parishioners and some priests."
text_b = "The waves of scandal angered not only victims' advocates but parishioners and some priests, to the point that Law could no longer run the archdiocese."
max_seq_length = 128

encoding = tokenizer(
    text_a,
    text_b,
    padding='max_length',
    truncation=True,
    max_length=max_seq_length)

print(encoding)