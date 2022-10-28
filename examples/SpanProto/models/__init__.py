# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 3:35 pm.
# @Author  : JianingWang
# @File    : __init__.py
# from models.chid_mlm import BertForChidMLM
# from models.duma import BertDUMAForMultipleChoice, AlbertDUMAForMultipleChoice, MegatronDumaForMultipleChoice
# from models.global_pointer import EffiGlobalPointer
# from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForMultipleChoice, BertTokenizer, \
    # AutoModelForQuestionAnswering
# from transformers.models.roformer import RoFormerTokenizer
from transformers.models.bert import BertTokenizerFast, BertForTokenClassification, BertTokenizer, BertForSequenceClassification


# from models.deberta import DebertaV2ForMultipleChoice, DebertaForMultipleChoice
# from models.fengshen.models.longformer import LongformerForMultipleChoice
# from models.kg import BertForPretrainWithKG, BertForPretrainWithKGV2
# from models.mlm.mlm_with_acc import BertForMaskedLMWithACC, RoFormerForMaskedLMWithACC
# from models.classification import build_cls_model
# from models.multiple_choice_tag import BertForTagMultipleChoice, RoFormerForTagMultipleChoice, MegatronBertForTagMultipleChoice
# from models.multiple_choice import MegatronBertForMultipleChoice, MegatronBertRDropForMultipleChoice
# from models.semeval7 import DebertaV2ForSemEval7MultiTask
# from roformer import RoFormerForTokenClassification, RoFormerForSequenceClassification

from models.span_proto import SpanProto

# task_type
MODEL_CLASSES = {
    'span_proto': SpanProto,
}

# model_type
TOKENIZER_CLASSES = {
    'bert': BertTokenizerFast,
    # 'wobert': RoFormerTokenizer,
    # 'roformer': RoFormerTokenizer,
    # 'bigbird': BertTokenizerFast,
    # 'erlangshen': BertTokenizerFast,
    # 'deberta': BertTokenizer,
    # 'roformer_v2': BertTokenizerFast
}
