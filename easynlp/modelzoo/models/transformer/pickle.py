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

"""
ParlAI's custom unpickler.

As modules move around or are renamed, it old torch model files become invalid,
since they look for modules in all the wrong places. Furthermore, we occasionally
use APEX for performance reasons, but we don't want to outright die if the user
has not installed it.

This module is to handle both of these issues. It is used like this:

>>> import parlai.utils.pickle
>>> state_dict = torch.load(filename, pickle_module=parlai.utils.pickle)
"""

import copy
import json
import pickle
import traceback
import os
import re
import pkg_resources
from abc import ABC, abstractmethod
from typing_extensions import final
from ...utils import logging
from .utils import warn_once

from typing import List, Dict, Optional, TypeVar, Any

class _Shared(Dict[str, Any]):
    """
    ParlAI ``shared`` Structure.

    The `shared` dict that is used to instantiate shared agents in ParlAI,
    e.g. when using batching, distributed training, etc.

    Type is ``TShared``.
    """

TShared = TypeVar('TShared', bound=_Shared)

# these keys are automatically removed upon save. This is a rather blunt hammer.
# It's preferred you indicate this at option definition time.
__AUTOCLEAN_KEYS__: List[str] = [
    "override",
    "batchindex",
    "download_path",
    "datapath",
    "verbose",
    # we don't save interactive mode or load from checkpoint, it's only decided by scripts or CLI
    "interactive_mode",
    "load_from_checkpoint",
]

try:
    from subword_nmt import learn_bpe, apply_bpe

    # Don't explicitly throw the runtime error unless the user needs it
    SUBWORD_BPE_INSTALLED = True
except ImportError:
    SUBWORD_BPE_INSTALLED = False

class FakeAPEXClass:
    pass

class Opt(dict):
    """
    Class for tracking options.

    Functions like a dict, but allows us to track the history of arguments as they are
    set.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []
        self.deepcopies = []

    def __setitem__(self, key, val):
        loc = traceback.format_stack(limit=2)[-2]
        self.history.append((key, val, loc))
        super().__setitem__(key, val)

    def __getstate__(self):
        return (self.history, self.deepcopies, dict(self))

    def __setstate__(self, state):
        self.history, self.deepcopies, data = state
        self.update(data)

    def __reduce__(self):
        return (Opt, (), self.__getstate__())

    def __deepcopy__(self, memo):
        """
        Override deepcopy so that history is copied over to new object.
        """
        # track location of deepcopy
        loc = traceback.format_stack(limit=3)[-3]
        self.deepcopies.append(loc)
        # copy all our children
        memo = Opt({k: copy.deepcopy(v) for k, v in self.items()})
        # deepcopy the history. history is only tuples, so we can do it shallow
        memo.history = copy.copy(self.history)
        # deepcopy the list of deepcopies. also shallow bc only strings
        memo.deepcopies = copy.copy(self.deepcopies)
        return memo

    def display_deepcopies(self):
        """
        Display all deepcopies.
        """
        if len(self.deepcopies) == 0:
            return 'No deepcopies performed on this opt.'
        return '\n'.join(f'{i}. {loc}' for i, loc in enumerate(self.deepcopies, 1))

    def display_history(self, key):
        """
        Display the history for an item in the dict.
        """
        changes = []
        i = 0
        for key_, val, loc in self.history:
            if key != key_:
                continue
            i += 1
            changes.append(f'{i}. {key} was set to {val} at:\n{loc}')
        if changes:
            return '\n'.join(changes)
        else:
            return f'No history for {key}'

    def save(self, filename: str) -> None:
        """
        Save the opt to disk.

        Attempts to 'clean up' any residual values automatically.
        """
        # start with a shallow copy
        dct = dict(self)

        # clean up some things we probably don't want to save
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dct, fp=f, indent=4)
            # extra newline for convenience of working with jq
            f.write('\n')

    @classmethod
    def load(cls, optfile: str):
        """
        Load an Opt from disk.
        """
        try:
            # try json first
            with open(optfile, 'r', encoding='utf-8') as t_handle:
                dct = json.load(t_handle)
        except UnicodeDecodeError:
            # oops it's pickled
            with open(optfile, 'rb') as b_handle:
                dct = pickle.load(b_handle)
        for key in __AUTOCLEAN_KEYS__:
            if key in dct:
                del dct[key]
        return cls(dct)

    @classmethod
    def load_init(cls, optfile: str):
        """
        Like load, but also looks in opt_presets folders.

        optfile may also be a comma-separated list of multiple presets/files.
        """
        if "," in optfile:
            # load and combine each of the individual files
            new_opt = cls()
            for subopt in optfile.split(","):
                new_opt.update(cls.load_init(subopt))
            return new_opt

        oa_filename = os.path.join("opt_presets", optfile + ".opt")
        user_filename = os.path.join(os.path.expanduser(f"~/.parlai"), oa_filename)
        if os.path.exists(optfile):
            return cls.load(optfile)
        elif os.path.exists(user_filename):
            # use a user's custom opt preset
            return cls.load(user_filename)
        else:
            # Maybe a bundled opt preset
            for root in ['parlai', 'parlai_internal', 'parlai_fb']:
                try:
                    if pkg_resources.resource_exists(root, oa_filename):
                        return cls.load(
                            pkg_resources.resource_filename(root, oa_filename)
                        )
                except ModuleNotFoundError:
                    continue

        # made it through without a return path so raise the error
        raise FileNotFoundError(
            f"Could not find filename '{optfile} or opt preset '{optfile}.opt'. "
            "Please check https://parl.ai/docs/opt_presets.html for a list "
            "of available opt presets."
        )

class BPEHelper(ABC):
    """
    Abstract BPE Helper.

    BPE Helper subclasses must implement appropriate abstractmethods.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        """
        Subclasses _should_ override __init__ to initialize other things.
        """
        self.lower = opt.get('dict_lower', False)
        self.maxtokens = opt.get('dict_maxtokens', -1)
        self.minfreq = opt.get('dict_minfreq', 0)

        self.opt = opt
        self.debug = opt.get('bpe_debug', False)
        self.add_prefix_space = opt.get('bpe_add_prefix_space', False)
        self._special_tokens: Dict[str, int] = {}
        self.bpe_dropout: Optional[float] = opt.get('bpe_dropout')
        self._bpe_dropout_enabled = False

    def enable_bpe_dropout(self, enabled: bool):
        """
        Used to toggle BPE dropout on (True) or off (False).
        """
        self._bpe_dropout_enabled = enabled

    @final
    def encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Checks for add_prefix_space; handles accordingly

        NOTE: DO NOT OVERRIDE

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """
        for special_token in self._special_tokens.keys():
            split = text.split(special_token)
            if len(split) > 1:
                output = []
                for i, piece in enumerate(split):
                    if i > 0:
                        output.append(special_token)
                    output += self.encode(piece)
                return output
        if self.add_prefix_space and not isinstance(self, HuggingFaceBpeHelper):
            text = f' {text}'
        return self.helper_encode(text)

    @abstractmethod
    def helper_encode(self, text: str) -> List[str]:
        """
        Tokenize text.

        Subclasses should override this method for encoding.

        :param text:
            text to tokenize

        :return tokens:
            A list of tokens
        """

    @final
    def decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str = ' '
    ) -> str:
        """
        Decode list of tokens into a text string.

        NOTE: DO NOT OVERRIDE

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        if self.debug:
            return delimiter.join(tokens)

        for i, token in enumerate(tokens):
            # note, HF ByteLevelBPE tokenizer handles special tokens itself in
            # a special way, so this will be skipped
            if token in self._special_tokens:
                # special token found. to the left, we've already cleared
                left = self.helper_decode(tokens[:i], token_ids[:i], delimiter)
                # token itself is easy to map to a string
                center = token
                # to the right, there may still be special tokens
                right = self.decode(
                    tokens[min(len(token_ids), i + 1) :],
                    token_ids[min(len(token_ids), i + 1) :],
                    delimiter,
                )
                return left + center + right

        # no special tokens found, we can fall back
        text = self.helper_decode(tokens, token_ids, delimiter)
        if self.add_prefix_space:
            assert text.startswith(' ')
            text = text.lstrip(' ')
        return text

    @abstractmethod
    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        Subclasses should override this method for decoding.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """

    @abstractmethod
    def sync_with_dict(self, dict_agent):
        """
        Sync BPE Helper dictionary with dict_agent dict.

        :param dict_agent:
            agent with which we are syncing the dictionary
        """

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        """
        Add special tokens to the tokenizer.

        These tokens are never split, and prioritized over the BPE tokenization.
        """
        # note, HF ByteLevelBPE tokenizer handles special tokens itself in
        # a special way, so this will be skipped
        for token in special_tokens:
            # exploiting dictionaries' insertion ordering to emulate ordered sets
            self._special_tokens[token] = 1

    def finalize(
        self, frequencies: Dict[str, int], num_symbols: int, minfreq: int
    ) -> bool:
        """
        Build the codecs.

        Default helpers are pre-trained and thus do not build their own codecs

        :param frequencies:
            dictionary of (token: frequency) pairs
        :param num_symbols:
            Number of BPE symbols. Recommend 30000-40000.  If <= 0, default
            30000 will be used.
        :param minfreq:
            Minimum frequency of a token before forced BPE decomposition. If <=
            0 will use subword-nmt default of 2.

        :return did_finalize:
            return whether codecs are finalized this call.
        """
        return False

    def copy_codecs_file(self, target_file: str):
        """
        Copy the codecs file to a new location.

        Default behavior is to do nothing.

        :param target_file:
            where to copy the codecs.
        """
        pass

    def should_sort(self) -> bool:
        """
        Return whether tokens should be sorted for this particular helper.

        DictionaryAgent sorts tokens upon saving; we don't generally want to sort with
        our pre-trained dictionaries, so default is False.
        """
        return False

class HuggingFaceBpeHelper(BPEHelper):
    """
    HuggingFace's ByteLevelBPE Tokenizer.

    Fast because Rust.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        super().__init__(opt, shared)
        # Default true for HF
        self.special_tok_map = {}  # map from HF
        self.add_prefix_space = opt.get('bpe_add_prefix_space', True)
        if self.add_prefix_space is None:
            self.add_prefix_space = True
        if opt.get('dict_loaded'):
            dfname = opt['dict_file']
            if os.path.exists(f'{dfname}-merges.txt'):
                opt['bpe_merge'] = f'{dfname}-merges.txt'
            if os.path.exists(f'{dfname}-vocab.json'):
                opt['bpe_vocab'] = f'{dfname}-vocab.json'
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                'Please install HuggingFace tokenizer with: pip install tokenizers'
            )

        if self.bpe_dropout:
            raise NotImplementedError(
                '--bpe-dropout is not supported with ByteLevelBPE because tokenizers '
                'library does not allow dynamically turning BPE on/off. You can use '
                '--dict-tokenizer slow_bytelevel_bpe to gain this feature.'
            )

        if self.lower:
            warn_once('Are you sure you want to lower case your BPE dictionary?')
        if self.maxtokens > 0 or self.minfreq > 0:
            raise ValueError(
                'You should not filter vocabulary with using --dict-tokenizer bytelevelbpe'
                ' (no --dict-minfreq or --dict-maxtokens).'
            )
        if 'bpe_vocab' not in opt:
            raise ValueError('--bpe-vocab is required for loading pretrained tokenizer')
        if 'bpe_merge' not in opt:
            raise ValueError('--bpe-merge is required for loading pretrained tokenizer')

        self.vocab_path = opt['bpe_vocab']
        self.merge_path = opt['bpe_merge']

        if not self.vocab_path or not self.merge_path:
            raise IOError(
                '--bpe-vocab and --bpe-merge are mandatory with '
                '--dict-tokenizer bytelevelbpe'
            )

        if not os.path.exists(self.vocab_path):
            raise IOError(
                f'File {self.vocab_path} does not exist. --bpe-vocab must be pretrained.'
            )
        if not os.path.exists(self.merge_path):
            raise IOError(
                f'File {self.merge_path} does not exist. --bpe-merge must be pretrained.'
            )

        self.tokenizer = ByteLevelBPETokenizer(
            self.vocab_path, self.merge_path, self.add_prefix_space
        )

    def helper_encode(self, text: str) -> List[str]:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        return self.tokenizer.encode(text).tokens

    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)

        return text

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        """
        Add special tokens to the tokenizer and dict_agent.
        """
        logging.debug(f'adding the following special tokens: {special_tokens}')
        self.tokenizer.add_special_tokens(special_tokens)  # add to HF

        for tok in special_tokens:
            parlai_key = dict_agent[tok]
            hf_key = self.tokenizer.token_to_id(tok)
            self.special_tok_map[parlai_key] = hf_key

    def sync_with_dict(self, dict_agent):
        """
        Sync the dictionary agent with Hugging Face tokenizer's BPE dict.

        Called only once on initialization.
        """
        special_tokens = [
            dict_agent.null_token,
            dict_agent.start_token,
            dict_agent.end_token,
            dict_agent.unk_token,
        ]
        self.add_special_tokens(dict_agent, special_tokens)

        for i in range(self.tokenizer.get_vocab_size() - len(special_tokens)):
            token = self.tokenizer.id_to_token(i)
            dict_agent.add_token(token)
            # We don't have access to the hugging face word frequency table,
            # just set it to 1 instead
            dict_agent.freq[token] = 1

    def save(self, dir_name: str, file_name: str):
        """
        Save appropriate files.

        :param dir_name:
            directory to save.
        :param file_name:
            file to save.
        """
        self.tokenizer.save_model(dir_name, file_name)

class SubwordBPEHelper(BPEHelper):
    """
    Helper class for performing BPE subword tokenization.

    For technical details, please refer to https://arxiv.org/abs/1508.07909.
    This class just wraps around the official subword-nmt repository.

    This API expects the user to call tokenize() (encode) onto the training data,
    then call finalize() to learn the encodings, and then iterate over the data
    in a second pass, calling tokenize() again to get processed output.
    """

    def __init__(self, opt: Opt, shared: TShared = None):
        """
        Initialize the BPE module.

        :param opt:
            options
        :param shared:
            shared dictionary
        """
        super().__init__(opt, shared)
        if not SUBWORD_BPE_INSTALLED:
            raise RuntimeError("Please run `pip install subword-nmt`")
        if not opt.get('dict_file'):
            raise RuntimeError('--dict-file is mandatory.')

        self.splitter = re.compile(r'\w+|[^\w\s]', re.UNICODE)

        self.codecs = f"{opt['dict_file']}.codecs"
        if os.path.exists(self.codecs):
            self._load_from_codecs()

    def add_special_tokens(self, dict_agent, special_tokens: List[str]):
        raise NotImplementedError(
            "--dict-tokenizer BPE does not support special tokens."
        )

    def helper_encode(self, text: str) -> List[str]:
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

        if hasattr(self, 'bpe'):
            return self.bpe.segment_tokens(tokens)
        else:
            return tokens

    def helper_decode(
        self, tokens: List[str], token_ids: List[int], delimiter: str
    ) -> str:
        """
        Decode list of tokens into text string.

        :param tokens:
            list of tokens
        :param token_ids:
            list of token ids
        :param delimiter:
            string delimiter for tokens

        :return text:
            decoded text
        """
        text = delimiter.join(tokens)
        text = text.replace('@@ ', '')
        # It's also possible that we get a BPE encoding on the end of the word
        if text.endswith('@@'):
            text = text[:-2]
        text = text.replace('__newln__', '\n')
        return text

    def finalize(
        self, frequencies: Dict[str, int], num_symbols: int = 30000, minfreq: int = 2
    ) -> bool:
        """
        Build the codecs.

        :param frequencies:
            dictionary of (token: frequency) pairs
        :param num_symbols:
            Number of BPE symbols. Recommend 30000-40000.  If <= 0, default
            30000 will be used.
        :param minfreq:
            Minimum frequency of a token before forced BPE decomposition. If <=
            0 will use subword-nmt default of 2.

        :return did_finalize:
            return whether codecs are finalized this call.
        """
        if hasattr(self, 'bpe'):
            # we already finalized the codecs
            return False

        logging.debug(f'Saving bpe codecs to {self.codecs}')

        dictionary = ("{} {}".format(k, v) for k, v in frequencies.items())

        if num_symbols <= 0:
            num_symbols = 30000
        if minfreq <= 0:
            minfreq = 2

        codec_dir, _ = os.path.split(self.codecs)
        os.makedirs(codec_dir, exist_ok=True)
        with open(self.codecs, 'w', encoding='utf-8') as outstream:
            learn_bpe.learn_bpe(
                dictionary,
                outstream,
                num_symbols=num_symbols,
                min_frequency=minfreq,
                is_dict=True,
            )

        self._load_from_codecs()
        return True

    def _load_from_codecs(self):
        """
        Load BPE from codecs file.
        """
        with open(self.codecs, 'r', encoding='utf-8') as codecs_file:
            self.bpe = apply_bpe.BPE(codecs_file)

    def copy_codecs_file(self, target_file: str):
        """
        Copy the codecs file to a new location.

        :param target_file:
            where to copy the codecs.
        """
        with open(target_file, 'w', encoding='utf-8') as wfile:
            with open(self.codecs, encoding='utf-8') as rfile:
                for line in rfile:
                    wfile.write(line)

    def sync_with_dict(self, dict_agent):
        """
        No need to sync subword BPE.
        """
        pass

    def should_sort(self) -> bool:
        """
        Return whether tokens should be sorted for this particular helper.

        We want to sort with SubwordBPEHelper.
        """
        return True


class Unpickler(pickle._Unpickler):  # type: ignore
    """
    Custom unpickler to handle moved classes and optional libraries.
    """

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith('apex.'):
                # user doesn't have apex installed. We'll deal with this later.
                return FakeAPEXClass
            else:
                if (
                    module == 'parlai.core.utils' or module == 'parlai.utils.misc'
                ) and name == 'Opt':
                    return Opt
                if module == 'parlai.core.dict' and name == '_BPEHelper':
                    return SubwordBPEHelper

                raise


def load(*args, **kwargs):
    return Unpickler(*args, **kwargs).load()
