import jieba
from .tokenization_bert import BertTokenizer
from .tokenization_albert import AlbertTokenizer
from .tokenization_gpt2 import GPT2Tokenizer


class Tokenizer(object):
    def __init__(self, backend="jieba", vocab_file=None, pretrain_model_name_or_path=None):
        self.backend = backend
        self.vocab_file = vocab_file
        if self.backend in ["bert", "gpt2", "albert"]:
            if self.backend == "bert":
                tokenizer_cls = BertTokenizer
            elif self.backend == "albert":
                tokenizer_cls = AlbertTokenizer
            elif self.tokenizer == "gpt2":
                tokenizer_cls = GPT2Tokenizer
            else:
                raise RuntimeError

            if vocab_file:
                self.tokenizer = tokenizer_cls(vocab_file=vocab_file)
            elif pretrain_model_name_or_path:
                self.tokenizer = tokenizer_cls.from_pretrained(
                    pretrained_model_name_or_path=pretrain_model_name_or_path)
        else:
            raise NotImplementedError("Tokenizer backend %s not implemented" % backend)

    def tokenize(self, text, *args, **kwargs):
        if self.backend == "jieba":
            return list(jieba.cut(text))
        elif self.backend == "bert":
            return self.tokenizer.tokenize(text, **kwargs)

    @property
    def vocab(self):
        if self.backend in ["bert", "gpt2"]:
            return self.tokenizer.vocab
        else:
            raise RuntimeError("Tokenizer backend %s does not have property `vocab`")

    @property
    def vocab_size(self):
        if self.backend in ["bert", "gpt2"]:
            return self.tokenizer.vocab_size
        else:
            raise RuntimeError("Tokenizer backend %s does not have property `vocab_size`")

    def convert_tokens_to_ids(self, tokens):
        if self.backend in ["bert", "gpt2"]:
            return self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            raise RuntimeError("Tokenizer backend %s does not have method `convert_tokens_to_ids`")


    def convert_ids_to_tokens(self, ids):
        if self.backend in ["bert", "gpt2"]:
            return self.tokenizer.convert_ids_to_tokens(ids)
        else:
            raise RuntimeError("Tokenizer backend %s does not have method `convert_tokens_to_ids`")
