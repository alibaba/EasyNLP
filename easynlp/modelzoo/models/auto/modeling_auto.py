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
""" Auto Model class. """


import warnings
from collections import OrderedDict

from ...utils import logging

# Add modeling imports here
from ..bert.modeling_bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertPreTrainedModel,
    BertModel,
)
from ..dkplm.modeling_dkplm import (
    DkplmForMaskedLM,
    DkplmForMultipleChoice,
    DkplmForNextSentencePrediction,
    DkplmForPreTraining,
    DkplmForQuestionAnswering,
    DkplmForSequenceClassification,
    DkplmForTokenClassification,
    DkplmLMHeadModel,
    DkplmPreTrainedModel,
    DkplmModel,
)

from ..megatron_bert.modeling_megatron_bert import (
    MegatronBertForMaskedLM,
    MegatronBertForMultipleChoice,
    MegatronBertForNextSentencePrediction,
    MegatronBertForPreTraining,
    MegatronBertForQuestionAnswering,
    MegatronBertForSequenceClassification,
    MegatronBertForTokenClassification,
    MegatronBertForCausalLM,
    MegatronBertPreTrainedModel,
    MegatronBertModel
)
# from ..clip.modeling_clip import CLIPModel
from ..gpt2.modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model

from ..pegasus.modeling_pegasus import PegasusForCausalLM, PegasusForConditionalGeneration, PegasusModel
from ..randeng.modeling_randeng import RandengForCausalLM, RandengForConditionalGeneration, RandengModel
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
)
from ..cnn.modeling_cnn import (
    TextCNNEncoder
)

from .auto_factory import auto_class_factory
from .configuration_auto import (
    BertConfig,
    DkplmConfig,
    GPT2Config,
    RobertaConfig,
    TextCNNConfig,
    KBertConfig,
    MegatronBertConfig,
    BartConfig,
    MT5Config,
    PegasusConfig,
    T5Config,
    BloomConfig,
    RandengConfig,
    KangarooConfig,
    TransformerConfig
)
from ..kbert.modeling_kbert import (
    KBertForMaskedLM,
    KBertForMultipleChoice,
    KBertForNextSentencePrediction,
    KBertForPreTraining,
    KBertForQuestionAnswering,
    KBertForSequenceClassification,
    KBertForTokenClassification,
    KBertLMHeadModel,
    KBertPreTrainedModel,
    KBertModel,
)

from ..bart.modeling_bart import (
    BartForCausalLM,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
    BartModel,
)

from ..kangaroo.modeling_kangaroo import (
    KangarooForMaskedLM,
    KangarooForMultipleChoice,
    KangarooForNextSentencePrediction,
    KangarooForPreTraining,
    KangarooForQuestionAnswering,
    KangarooForSequenceClassification,
    KangarooForTokenClassification,
    KangarooLMHeadModel,
    KangarooPreTrainedModel,
    KangarooModel,
)

from ..mt5.modeling_mt5 import MT5ForConditionalGeneration, MT5Model
from ..pegasus.modeling_pegasus import PegasusForCausalLM, PegasusForConditionalGeneration, PegasusModel
from ..t5.modeling_t5 import T5ForConditionalGeneration, T5Model
from ..bloom.modeling_bloom import BloomForCausalLM, BloomModel, BloomForTokenClassification, BloomForSequenceClassification, BloomPreTrainedModel
from ..transformer.modeling_transformer import TransformerModel

logger = logging.get_logger(__name__)

PRETRAINED_MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (BertConfig, BertPreTrainedModel),
        (DkplmConfig, DkplmPreTrainedModel),
        (MegatronBertConfig, MegatronBertPreTrainedModel),
        (KBertConfig, KBertPreTrainedModel),
        (KangarooConfig, KangarooPreTrainedModel)
    ]
)

MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (RobertaConfig, RobertaModel),
        # (LayoutLMConfig, LayoutLMModel),
        # (SqueezeBertConfig, SqueezeBertModel),
        (BertConfig, BertModel),
        (DkplmConfig, DkplmModel),
        (MegatronBertConfig, MegatronBertModel),
        (GPT2Config, GPT2Model),
        (TextCNNConfig, TextCNNEncoder),
        (KBertConfig, KBertModel),
        (KangarooConfig, KangarooModel),
        (BartConfig, BartModel),
        (MT5Config, MT5Model),
        (T5Config, T5Model),
        (PegasusConfig, PegasusModel),
        (BloomConfig, BloomModel),
        (RandengConfig, RandengModel),
        (TransformerConfig, TransformerModel)
    ]
)

MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        # Model for pre-training mapping
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForPreTraining),
        (DkplmConfig, DkplmForPreTraining),
        (GPT2Config, GPT2LMHeadModel),
        (MegatronBertConfig, MegatronBertForPreTraining),
        (KBertConfig, KBertForPreTraining),
        (T5Config, T5ForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (BloomConfig, BloomPreTrainedModel),
        (KangarooConfig, KangarooForPreTraining)
    ]
)

MODEL_WITH_LM_HEAD_MAPPING = OrderedDict(
    [
        # Model with LM heads mapping
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (DkplmConfig, DkplmForMaskedLM),
        (MegatronBertConfig, MegatronBertForMaskedLM),
        (GPT2Config, GPT2LMHeadModel),
        (KBertConfig, KBertForMaskedLM),
        (T5Config, T5ForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (KangarooConfig, KangarooForMaskedLM)
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING = OrderedDict(
    [
        # Model for Causal LM mapping
        (RobertaConfig, RobertaForCausalLM),
        (BertConfig, BertLMHeadModel),
        (DkplmConfig, DkplmLMHeadModel),
        (MegatronBertConfig, MegatronBertForCausalLM),
        (GPT2Config, GPT2LMHeadModel),
        (KBertConfig, KBertLMHeadModel),
        (BartConfig, BartForCausalLM),
        (PegasusConfig, PegasusForCausalLM),
        (BloomConfig, BloomForCausalLM),
        (RandengConfig, RandengForCausalLM),
        (KangarooConfig, KangarooLMHeadModel)
    ]
)

MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (RobertaConfig, RobertaForMaskedLM),
        (BertConfig, BertForMaskedLM),
        (DkplmConfig, DkplmForMaskedLM),
        (MegatronBertConfig, MegatronBertForMaskedLM),
        (KBertConfig, KBertForMaskedLM),
        (BartConfig, BartForConditionalGeneration),
        (KangarooConfig, KangarooForMaskedLM)
    ]
)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = OrderedDict(
    [
        (MT5Config, MT5ForConditionalGeneration),
        (T5Config, T5ForConditionalGeneration),
        (PegasusConfig, PegasusForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration),
        (BloomConfig, BloomForCausalLM),
        (RandengConfig, RandengForConditionalGeneration),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (RobertaConfig, RobertaForSequenceClassification),
        (BertConfig, BertForSequenceClassification),
        (MegatronBertConfig, MegatronBertForSequenceClassification),
        (DkplmConfig, DkplmForSequenceClassification),
        (GPT2Config, GPT2ForSequenceClassification),
        (KBertConfig, KBertForSequenceClassification),
        (BloomConfig, BloomForSequenceClassification),
        (KangarooConfig, KangarooForSequenceClassification)
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (RobertaConfig, RobertaForQuestionAnswering),
        (BertConfig, BertForQuestionAnswering),
        (DkplmConfig, DkplmForQuestionAnswering),
        (MegatronBertConfig, MegatronBertForQuestionAnswering),
        (KBertConfig, KBertForQuestionAnswering),
        (BartConfig, BartForQuestionAnswering),
        (KangarooConfig, KangarooForQuestionAnswering)
    ]
)


MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (RobertaConfig, RobertaForTokenClassification),
        (BertConfig, BertForTokenClassification),
        (DkplmConfig, DkplmForTokenClassification),
        (MegatronBertConfig, MegatronBertForTokenClassification),
        (KBertConfig, KBertForTokenClassification),
        (BloomConfig, BloomForTokenClassification),
        (KangarooConfig, KangarooForTokenClassification)
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (RobertaConfig, RobertaForMultipleChoice),
        (BertConfig, BertForMultipleChoice),
        (DkplmConfig, DkplmForMultipleChoice),
        (MegatronBertConfig, MegatronBertForMultipleChoice),
        (KBertConfig, KBertForTokenClassification),
        (BloomConfig, BloomForTokenClassification),
        (KangarooConfig, KangarooForTokenClassification)
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, BertForNextSentencePrediction),
        (DkplmConfig, DkplmForNextSentencePrediction),
        (MegatronBertConfig, MegatronBertForNextSentencePrediction),
        (KBertConfig, KBertForNextSentencePrediction),
        (KangarooConfig, KangarooForNextSentencePrediction)
    ]
)


AutoModel = auto_class_factory("AutoModel", MODEL_MAPPING)

AutoPreTrainedModel = auto_class_factory("AutoPreTrainedModel", PRETRAINED_MODEL_MAPPING)

AutoModelForPreTraining = auto_class_factory(
    "AutoModelForPreTraining", MODEL_FOR_PRETRAINING_MAPPING, head_doc="pretraining"
)

# Private on purpose, the public class will add the deprecation warnings.
_AutoModelWithLMHead = auto_class_factory(
    "AutoModelWithLMHead", MODEL_WITH_LM_HEAD_MAPPING, head_doc="language modeling"
)

AutoModelForCausalLM = auto_class_factory(
    "AutoModelForCausalLM", MODEL_FOR_CAUSAL_LM_MAPPING, head_doc="causal language modeling"
)

AutoModelForMaskedLM = auto_class_factory(
    "AutoModelForMaskedLM", MODEL_FOR_MASKED_LM_MAPPING, head_doc="masked language modeling"
)

AutoModelForSeq2SeqLM = auto_class_factory(
    "AutoModelForSeq2SeqLM",
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="t5-base",
)

AutoModelForSequenceClassification = auto_class_factory(
    "AutoModelForSequenceClassification", MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, head_doc="sequence classification"
)

AutoModelForQuestionAnswering = auto_class_factory(
    "AutoModelForQuestionAnswering", MODEL_FOR_QUESTION_ANSWERING_MAPPING, head_doc="question answering"
)

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForMultipleChoice = auto_class_factory(
    "AutoModelForMultipleChoice", MODEL_FOR_MULTIPLE_CHOICE_MAPPING, head_doc="multiple choice"
)

AutoModelForNextSentencePrediction = auto_class_factory(
    "AutoModelForNextSentencePrediction",
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    head_doc="next sentence prediction",
)


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
