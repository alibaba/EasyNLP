# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team and Alibaba PAI team
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

from typing import TYPE_CHECKING

from ...file_utils import (
    _BaseLazyModule,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_bert": ["BertConfig"],
    "tokenization_bert": ["BasicTokenizer", "BertTokenizer", "WordpieceTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_bert_fast"] = ["BertTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bert"] = [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertForMaskedLM",
        "BertForMultipleChoice",
        "BertForNextSentencePrediction",
        "BertForPreTraining",
        "BertForQuestionAnswering",
        "BertForSequenceClassification",
        "BertForTokenClassification",
        "BertLayer",
        "BertLMHeadModel",
        "BertModel",
        "BertPreTrainedModel",
    ]

if TYPE_CHECKING:
    # from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
    from .configuration_bert import BertConfig
    from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer

    if is_tokenizers_available():
        from .tokenization_bert_fast import BertTokenizerFast

    if is_torch_available():
        from .modeling_bert import (
            # BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLayer,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            # load_tf_weights_in_bert,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
