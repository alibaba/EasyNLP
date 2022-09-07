import math

from ...utils import get_pretrain_model_path
#from ....modelzoo import AutoTokenizer, BertTokenizer, GPT2Tokenizer
from ...modelzoo import AutoTokenizer

class ImageTextBERTTokenizer(object):
    def __init__(self, pretrained_model_name_or_path, start_id, unk_token="[UNK]", end_token="[PAD]"):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = get_pretrain_model_path('bert-base-chinese')

        #self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
        print (f"tokenizer load from {pretrained_model_name_or_path}")

        max_num = -math.inf
        for key in self.tokenizer.vocab.keys():
            num = self.tokenizer.vocab[key]
            self.tokenizer.vocab[key] = num + start_id
            if max_num < num:
                max_num = num
        
        if unk_token not in self.tokenizer.vocab:
            max_num += 1
            self.tokenizer.vocab[unk_token] = max_num
        if end_token not in self.tokenizer.vocab:
            max_num += 1
            self.tokenizer.vocab[end_token] = max_num
        
        self.tokenizer.ids_to_tokens = {v: k for k, v in self.tokenizer.vocab.items()}
        self.encoder = self.tokenizer.vocab
        self.decoder = self.tokenizer.ids_to_tokens

        self.unk_token = unk_token
        self.end_token = end_token
        self.unk_token_id = self.encoder[self.unk_token]
        self.end_token_id = self.encoder[self.end_token]
        #self.unk_token_id = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        #self.end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_token)
    
    def encode(self, text):
        return self.tokenizer.encode(text)[1:-1]
    
    def decode(self, token_list):
        return "".join(self.tokenizer.decode(token_list).split())

    def __len__(self):
        return len(self.tokenizer)

class ImageTextGPT2Tokenizer(object):
    def __init__(self, pretrained_model_name_or_path, start_id, unk_token="<|endoftext|>", end_token="<|endoftext|>"):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = get_pretrain_model_path('gpt2')

        #self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=True)
        print (f"tokenizer load from {pretrained_model_name_or_path}")
        
        max_num = -math.inf
        for key in self.tokenizer.encoder.keys():
            num = self.tokenizer.encoder[key]
            self.tokenizer.encoder[key] = num + start_id
            if max_num < num:
                max_num = num
        
        if unk_token not in self.tokenizer.encoder:
            max_num += 1
            self.tokenizer.encoder[unk_token] = max_num
        if end_token not in self.tokenizer.encoder:
            max_num += 1
            self.tokenizer.encoder[end_token] = max_num
        
        self.tokenizer.decoder = {v: k for k, v in self.tokenizer.encoder.items()}
        self.encoder = self.tokenizer.encoder
        self.decoder = self.tokenizer.decoder

        self.unk_token = unk_token
        self.end_token = end_token
        self.unk_token_id = self.encoder[self.unk_token]
        self.end_token_id = self.encoder[self.end_token]
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, token_list):
        return self.tokenizer.decode(token_list)


