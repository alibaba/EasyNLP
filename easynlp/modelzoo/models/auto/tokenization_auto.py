# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team and Alibaba PAI Team.
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
""" Auto Tokenizer class. """

import imp
import json
import os
from collections import OrderedDict
from typing import Dict, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...file_utils import (
    cached_path,
    hf_bucket_url,
    is_offline_mode,
    is_sentencepiece_available,
    is_tokenizers_available,
)
from ...tokenization_utils_base import TOKENIZER_CONFIG_FILE
from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer
from ..dkplm.tokenization_dkplm import DkplmTokenizer
from ..megatron_bert.tokenization_megatron_bert import MegatronBertTokenizer
from ..gpt2.tokenization_gpt2 import GPT2Tokenizer
from ..roberta.tokenization_roberta import RobertaTokenizer
from ..cnn.tokenization_cnn import TextCNNTokenizer
from ..kbert.tokenization_kbert import KBertTokenizer
from ..bart.tokenization_bart import BartTokenizer
from ..bloom.tokenization_bloom_fast import BloomTokenizerFast
from ..randeng.tokenization_randeng import RandengTokenizer
from ..kangaroo.tokenization_kangaroo import KangarooTokenizer
from ..transformer.tokenization_transformer import TransformerTokenizer

from .configuration_auto import (
    AutoConfig,
    DkplmConfig,
    MegatronBertConfig,
    BertConfig,
    GPT2Config,
    RobertaConfig,
    replace_list_option_in_docstrings,
    TextCNNConfig,
    ARTISTConfig,
    MinGPTI2TConfig,
    KBertConfig,
    BartConfig,
    MT5Config,
    PegasusConfig,
    T5Config,
    BloomConfig,
    RandengConfig,
    KangarooConfig,
    TransformerConfig
)

if is_sentencepiece_available():
    from ..mt5 import MT5Tokenizer
    from ..pegasus.tokenization_pegasus import PegasusTokenizer
    from ..t5.tokenization_t5 import T5Tokenizer
else:
    MT5Tokenizer = None
    PegasusTokenizer = None
    T5Tokenizer = None

if is_tokenizers_available():
    from ...tokenization_utils_fast import PreTrainedTokenizerFast
    from ..bert.tokenization_bert_fast import BertTokenizerFast
    from ..dkplm.tokenization_dkplm_fast import DkplmTokenizerFast
    from ..megatron_bert.tokenization_modeling_bert_fast import MegatronBertTokenizerFast
    from ..gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
    from ..kbert.tokenization_kbert_fast import KBertTokenizerFast
    from ..bart.tokenization_bart_fast import BartTokenizerFast
    from ..mt5 import MT5TokenizerFast
    from ..pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
    from ..t5.tokenization_t5_fast import T5TokenizerFast
    from ..bloom.tokenization_bloom_fast import BloomTokenizerFast
    from ..kangaroo.tokenization_kangaroo_fast import KangarooTokenizerFast
else:
    BertTokenizerFast = None
    GPT2TokenizerFast = None
    RobertaTokenizerFast = None
    DkplmTokenizerFast = None
    MegatronBertTokenizerFast = None
    PreTrainedTokenizerFast = None
    KBertTokenizerFast = None
    BartTokenizerFast = None
    PegasusTokenizerFast = None
    T5TokenizerFast = None
    BloomTokenizerFast = None
    KangarooTokenizerFast = None


logger = logging.get_logger(__name__)


TOKENIZER_MAPPING = OrderedDict(
    [
        (RobertaConfig, (RobertaTokenizer, RobertaTokenizerFast)),
        (BertConfig, (BertTokenizer, BertTokenizerFast)),
        (DkplmConfig, (DkplmTokenizer, DkplmTokenizerFast)),
        (MegatronBertConfig, (MegatronBertTokenizer, MegatronBertTokenizerFast)),
        (GPT2Config, (GPT2Tokenizer, GPT2TokenizerFast)),
        (TextCNNConfig, (TextCNNTokenizer, None)),
        (ARTISTConfig, (BertTokenizer, BertTokenizerFast)),
        (MinGPTI2TConfig, (BertTokenizer, BertTokenizerFast)),
        (KBertConfig, (KBertTokenizer, KBertTokenizerFast)),
        (T5Config, (T5Tokenizer, T5TokenizerFast)),
        (MT5Config, (MT5Tokenizer, MT5TokenizerFast)),
        (PegasusConfig, (PegasusTokenizer, PegasusTokenizerFast)),
        (BartConfig, (BartTokenizer, BartTokenizerFast)),
        (BloomConfig, (None, BloomTokenizerFast)),
        (RandengConfig, (RandengTokenizer, None)),
        (KangarooConfig, (KangarooTokenizer, KangarooTokenizerFast)),
        (TransformerConfig, (TransformerTokenizer, None))
    ]
)

# For tokenizers which are not directly mapped from a config
NO_CONFIG_TOKENIZER = [
    PreTrainedTokenizerFast,
]


SLOW_TOKENIZER_MAPPING = {
    k: (v[0] if v[0] is not None else v[1])
    for k, v in TOKENIZER_MAPPING.items()
    if (v[0] is not None or v[1] is not None)
}


def tokenizer_class_from_name(class_name: str):
    all_tokenizer_classes = (
        [v[0] for v in TOKENIZER_MAPPING.values() if v[0] is not None]
        + [v[1] for v in TOKENIZER_MAPPING.values() if v[1] is not None]
        + [v for v in NO_CONFIG_TOKENIZER if v is not None]
    )
    for c in all_tokenizer_classes:
        if c.__name__ == class_name:
            return c


def get_tokenizer_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
            This can be either:

            - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
              namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
            - a path to a `directory` containing a configuration file saved using the
              :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g., ``./my_model_directory/``.

        cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (:obj:`Dict[str, str]`, `optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        local_files_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, will only try to load the tokenizer configuration from local files.

    .. note::

        Passing :obj:`use_auth_token=True` is required when you want to use a private model.


    Returns:
        :obj:`Dict`: The configuration of the tokenizer.

    """
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        config_file = os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_FILE)
    else:
        config_file = hf_bucket_url(
            pretrained_model_name_or_path, filename=TOKENIZER_CONFIG_FILE, revision=revision, mirror=None
        )

    try:
        # Load from URL or cache if already cached
        resolved_config_file = cached_path(
            config_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
        )

    except EnvironmentError:
        logger.info("Could not locate the tokenizer configuration file, will try to use the model config instead.")
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the :meth:`AutoTokenizer.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                      using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__()`` method.
            config (:class:`~transformers.PreTrainedConfig`, `optional`)
                The configuration object used to determine the tokenizer class to instantiate.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to try to load the fast version of the tokenizer.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__()`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__()`` for more details.

        """
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True

        use_fast = kwargs.pop("use_fast", False)

        # First, let's try to use the tokenizer_config file to get the tokenizer class.
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        # If that did not work, let's try to use the config.
        if config_tokenizer_class is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            config_tokenizer_class = config.tokenizer_class

        # If we have the tokenizer class from the tokenizer config or the model config we're good!
        if config_tokenizer_class is not None:
            tokenizer_class = None
            if use_fast and not config_tokenizer_class.endswith("Fast"):
                tokenizer_class_candidate = f"{config_tokenizer_class}Fast"
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)
            if tokenizer_class is None:
                tokenizer_class_candidate = config_tokenizer_class
                tokenizer_class = tokenizer_class_from_name(tokenizer_class_candidate)

            if tokenizer_class is None:
                raise ValueError(
                    f"Tokenizer class {tokenizer_class_candidate} does not exist or is not currently imported."
                )
            return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

        if type(config) in TOKENIZER_MAPPING.keys():
            tokenizer_class_py, tokenizer_class_fast = TOKENIZER_MAPPING[type(config)]
            if tokenizer_class_fast and (use_fast or tokenizer_class_py is None):
                return tokenizer_class_fast.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                if tokenizer_class_py is not None:
                    return tokenizer_class_py.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
                else:
                    raise ValueError(
                        "This tokenizer cannot be instantiated. Please make sure you have `sentencepiece` installed "
                        "in order to use this tokenizer."
                    )

        raise ValueError(
            f"Unrecognized configuration class {config.__class__} to build an AutoTokenizer.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in TOKENIZER_MAPPING.keys())}."
        )
