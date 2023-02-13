# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization classes for BlenderBot """

from typing import List
from ...tokenization_utils import PreTrainedTokenizer
import regex as re
import collections
import os
import sys
import random
from tokenizers import ByteLevelBPETokenizer
from easynlp.utils import get_pretrain_model_path

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.txt",
    "codecs_file": "dict.codecs"
}
BYTELEVEL_VOCAB_FILE_NAME = "vocab.json"

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def unescape(s):
    r"""
    Revert escaped characters back to their special version.

    For example, \\n => \n and \\t => \t

    :param s:
        string to unescape
    """
    return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    for index, line in enumerate(lines):
        split = line.rstrip("\n\r").split("\t")
        token = unescape(split[0])
        vocab[token] = index
    return vocab

def load_codecs(codecs_file, merges=-1):
    """Loads a bpe codecs file into a dictionary."""
    bpe_codes = dict()
    bpe_codes_reverse = dict()
    version = None
    with open(codecs_file, 'r', encoding='utf-8') as codecs_file:
        codecs_file.seek(0)
        offset=1

        # check version information
        firstline = codecs_file.readline()
        if firstline.startswith('#version:'):
            version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            version = (0, 1)
            codecs_file.seek(0)

        bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codecs_file.read().rstrip('\n').split('\n')) if (n < merges or merges == -1)]

        for i, item in enumerate(bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(bpe_codes)))])

        bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in bpe_codes.items()])
    return bpe_codes, bpe_codes_reverse, version

def find_ngrams(token_dict, text, n):
    """
    Break text into ngrams that appear in ``token_dict``.

    :param token_dict:
        ``dict`` to check for ngrams
    :param text:
        ``str`` to look for ngrams in
    :param n:
        ``int`` max size of ngrams
    """
    # base case
    if n <= 1:
        return text
    # tokens committed to output
    saved_tokens = []
    # tokens remaining to be searched in sentence
    search_tokens = text[:]
    # tokens stored until next ngram found
    next_search = []
    while len(search_tokens) >= n:
        ngram = ' '.join(search_tokens[:n])
        if ngram in token_dict:
            # first, search previous unmatched words for smaller ngrams
            sub_n = min(len(next_search), n - 1)
            saved_tokens.extend(find_ngrams(token_dict, next_search, sub_n))
            next_search.clear()
            # then add this ngram
            saved_tokens.append(ngram)
            # then pop this ngram from the remaining words to search
            search_tokens = search_tokens[n:]
        else:
            next_search.append(search_tokens.pop(0))
    remainder = next_search + search_tokens
    sub_n = min(len(remainder), n - 1)
    saved_tokens.extend(find_ngrams(token_dict, remainder, sub_n))
    return saved_tokens

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out

class TransformerTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        codecs_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        null_token='__null__',
        bos_token='__start__',
        eos_token='__end__',
        unk_token='__unk__',
        max_ngram_size=-1,
        max_tokens=-1,
        tokenizer='bpe',
        separator='@@',
        **kwargs
    ):
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            pad_token=null_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            max_ngram_size=max_ngram_size,
            max_tokens=max_tokens,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.max_ngram_size = max_ngram_size
        self.max_tokens = max_tokens
        self.null_token = null_token
        self.tokenizer = tokenizer
        self.separator = separator


        # Search vocab and codecs file under pretrain model path
        # when they are not found under checkpoint dir
        if not vocab_file or not os.path.isfile(vocab_file):
            pretrain_model_path = get_pretrain_model_path(kwargs.get('origin_model_name', ''))
            vocab_file = os.path.join(pretrain_model_path, VOCAB_FILES_NAMES['vocab_file'])
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained "
                    "model use `tokenizer = TransformerTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
        if not codecs_file or not os.path.isfile(codecs_file):
            pretrain_model_path = get_pretrain_model_path(kwargs.get('origin_model_name', ''))
            codecs_file = os.path.join(pretrain_model_path, VOCAB_FILES_NAMES['codecs_file'])
            if not os.path.isfile(codecs_file):
                raise ValueError(
                    f"Can't find a bpe codecs file at path '{codecs_file}'. To load the codecs from a Google pretrained "
                    "model use `tokenizer = TransformerTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.codecs, self.codecs_reverse, self.version = load_codecs(codecs_file)

        self.splitter = re.compile(r'\w+|[^\w\s]', re.UNICODE)
        self.cache = {}

        if self.tokenizer == 'bytelevelbpe':
            vocab_path = os.path.join(os.path.dirname(vocab_file), BYTELEVEL_VOCAB_FILE_NAME)
            assert os.path.exists(vocab_path), "Vocab file for bytelevel BPE not found! please check and retry."

            self.help_tokenizer = ByteLevelBPETokenizer(
                vocab_path, codecs_file, True
            )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def encode(self, orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, dropout=0):
        """Encode word based on list of BPE merge operations, which are applied consecutively"""
        if not dropout and orig in cache:
            return cache[orig]

        if len(orig) == 1:
            return orig

        if version == (0, 1):
            word = list(orig) + ['</w>']
        elif version == (0, 2): # more consistent handling of word-final segments
            word = list(orig[:-1]) + [orig[-1] + '</w>']
        else:
            raise NotImplementedError

        while len(word) > 1:

            # get list of symbol pairs; optionally apply dropout
            pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random.random() > dropout) and pair in bpe_codes]

            if not pairs:
                break

            #get first merge operation in list of BPE codes
            bigram = min(pairs)[2]

            # find start position of all pairs that we want to merge
            positions = [i for (rank,i,pair) in pairs if pair == bigram]

            i = 0
            new_word = []
            bigram = ''.join(bigram)
            for j in positions:
                # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
                if j < i:
                    continue
                new_word.extend(word[i:j]) # all symbols before merged pair
                new_word.append(bigram) # merged pair
                i = j+2 # continue after merged pair
            new_word.extend(word[i:]) # add all symbols until end of word
            word = new_word

        # don't print end-of-word symbols
        if word[-1] == '</w>':
            word = word[:-1]
        elif word[-1].endswith('</w>'):
            word[-1] = word[-1][:-4]

        word = tuple(word)
        if vocab:
            word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

        cache[orig] = word
        return word
        
    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in [word]
                        for out in self.encode(segment,
                                          self.codecs,
                                          self.codecs_reverse,
                                          None,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def bpe_tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text with bpe if codecs are already finalized.

        Otherwise, returns the regularly split tokens that will train the bpe.

        :param text:
            Raw text to tokenize.
        :return:
            a list of tokens. Will use BPE once finalized.
        """
        text = text.replace('\n', ' __newln__ ')
        tokens = self.splitter.findall(text)

        if self.tokenizer == 'bpe':
            return self.segment_tokens(tokens)
        else:
            return tokens

    def _tokenize(self, text: str):
        """
        Return a sequence of tokens from the iterable.

        Also handles special tokens for some tokenizers
        """
        if self.tokenizer in ('re', 'split', 'space'):
            for special_token in self.additional_special_tokens:
                index = text.find(special_token)
                if index == -1:
                    continue
                left = text[:index]
                right = text[index + len(special_token) :]
                tokens_left = self._tokenize(left) if left else []
                tokens_right = self._tokenize(right) if right else []
                return tokens_left + [special_token] + tokens_right

        if self.do_lower_case:
            text = text.lower()

        # calls the selected tokenizer function e.g. 're' => re_tokenize(text)
        if self.tokenizer == 'bytelevelbpe':
            word_tokens = self.help_tokenizer.encode(text).tokens
        else:
            word_tokens = self.bpe_tokenize(text)

        if self.max_ngram_size > 1:
            # search for ngrams during parse-time
            # TODO(ahm): support build-time ngrams using word2vec heuristic?
            word_tokens = find_ngrams(self.tok2ind, word_tokens, self.max_ngram_size)
        return word_tokens
    
    def _decode(self, **kwargs) -> str:
        if self.tokenizer == 'bytelevelbpe':
            token_ids = kwargs.get('token_ids')
            extra_tokens = 4
            token_ids = [idx-extra_tokens for idx in token_ids]
            text = self.help_tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            text = super()._decode(**kwargs)

            text = text.replace('@@ ', '')
            if text.endswith('@@'):
                text = text[:-2]
        
        text = text.replace('__newln__', '\n')
        return text
    
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        idx = self.vocab.get(token, self.vocab.get(self.unk_token))
        # if self.tokenizer == 'bytelevelbpe':
        #     extra_tokens = 4  # length of special tokens
        #     idx -= extra_tokens
        return idx

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.ids_to_tokens.get(index, self.unk_token)
        token = '\n' if token == '__newln__' else token
        return token