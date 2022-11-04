# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 4:49 pm.
# @Author  : JianingWang
# @File    : __init__.py
from processor.few_ner.data_process import FewNERDProcessor, CrossNERProcessor

processor_map = {
    'fewnerd': FewNERDProcessor,
    'crossner': CrossNERProcessor
}
