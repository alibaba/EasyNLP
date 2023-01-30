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
""" Auto Config class. """

import re
from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ..bert.configuration_bert import BertConfig
from ..dkplm.configuration_dkplm import DkplmConfig
from ..megatron_bert.configuration_megatron_bert import MegatronBertConfig
from ..gpt2.configuration_gpt2 import GPT2Config

from ..roberta.configuration_roberta import RobertaConfig
from ..cnn.configuration_cnn import TextCNNConfig
from ..artist.configuration_artist import ARTISTConfig
from ..mingpt_i2t.configuration_mingpt_i2t import MinGPTI2TConfig
from ..clip.configuration_clip import CLIPConfig
from ..kbert.configuration_kbert import KBertConfig
from ..kangaroo.configuration_kangaroo import KangarooConfig

from ..bart.configuration_bart import BartConfig
from ..mt5.configuration_mt5 import MT5Config
from ..pegasus.configuration_pegasus import PegasusConfig
from ..t5.configuration_t5 import T5Config
from ..bloom.configuration_bloom import BloomConfig
from ..randeng.configuration_randeng import RandengConfig
from ..transformer.configuration_transformer import TransformerConfig

CONFIG_MAPPING = OrderedDict(
    [
        ("roberta", RobertaConfig),
        ("bert", BertConfig),
        ("dkplm", DkplmConfig),
        ("megatron-bert", MegatronBertConfig),
        ("gpt2", GPT2Config),
        ("cnn", TextCNNConfig),
        ("artist", ARTISTConfig),
        ("artist_i2t", MinGPTI2TConfig),
        ("mingpt_i2t", MinGPTI2TConfig),
        ("clip", CLIPConfig),
        ("kbert", KBertConfig),
        ("mt5", MT5Config),
        ("t5", T5Config),
        ("pegasus", PegasusConfig),
        ("bart", BartConfig),
        ("bloom", BloomConfig),
        ("randeng", RandengConfig),
        ("kangaroo", KangarooConfig),
        ("transformer", TransformerConfig)
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        ("roberta", "RoBERTa"),
        ("bert", "BERT"),
        ("dkplm", "DKPLM"),
        ("megatron-bert", "MEGATRON-BERT"),
        ("gpt2", "OpenAI GPT-2"),
        ("cnn", 'TextCNN'),
        ("artist", "ARTIST"),
        ("artist_i2t", "MinGPTI2TConfig"),
        ("mingpt_i2t", "MinGPTI2TConfig"), 
        ("clip", "CLIP"),
        ("kbert", "KBERT"),
        ("t5", "T5"),
        ("pegasus", "Pegasus"),
        ("bart", "BART"),
        ("mt5", "mT5"),
        ("bloom", "Bloom"),
        ("randeng", "Randeng"),
        ("kangaroo", "KANGAROO"),
        ("transformer", "Transformer")
    ]
)


def _get_class_name(model_class):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f":class:`~transformers.{c.__name__}`" for c in model_class])
    return f":class:`~transformers.{model_class.__name__}`"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {
                model_type: f":class:`~transformers.{config.__name__}`"
                for model_type, config in CONFIG_MAPPING.items()
            }
        else:
            model_type_to_name = {
                model_type: _get_class_name(config_to_class[config])
                for model_type, config in CONFIG_MAPPING.items()
                if config in config_to_class
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {config.__name__: _get_class_name(clas) for config, clas in config_to_class.items()}
        config_to_model_name = {
            config.__name__: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING.items()
        }
        lines = [
            f"{indent}- :class:`~transformers.{config_name}` configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the :meth:`~transformers.AutoConfig.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the :obj:`model_type` property of the config object
        that is loaded, or when it's missing, by falling back to using pattern matching on
        :obj:`pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model configuration hosted inside a model repo on
                      huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                      namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing a configuration file saved using the
                      :meth:`~transformers.PretrainedConfig.save_pretrained` method, or the
                      :meth:`~transformers.PreTrainedModel.save_pretrained` method, e.g., ``./my_model_directory/``.
                    - A path or url to a saved configuration JSON `file`, e.g.,
                      ``./my_model_directory/configuration.json``.
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
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs(additional keyword arguments, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        """
        kwargs["_from_auto"] = True
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict:
            config_class = CONFIG_MAPPING[config_dict["model_type"]]
            return config_class.from_dict(config_dict, **kwargs)
        else:
            # Fallback: use pattern matching on the string.
            for pattern, config_class in CONFIG_MAPPING.items():
                if pattern in str(pretrained_model_name_or_path):
                    return config_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            "Should have a `model_type` key in its config.json, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )
