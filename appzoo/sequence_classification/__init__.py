# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
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

# from typing import TYPE_CHECKING
# from ...modelzoo.file_utils import _BaseLazyModule

# _import_structure = {
#     "data": ['ClassificationDataset', 
#             'DistillatoryClassificationDataset', 
#             'FewshotSequenceClassificationDataset'],
#     "evaluator": ['SequenceClassificationEvaluator', 
#                   'SequenceMultiLabelClassificationEvaluator'],
#     "model": ['SequenceClassification', 
#               'SequenceMultiLabelClassification', 
#               'FewshotClassification', 
#               'CPTClassification', 
#               'DistillatoryBaseApplication'],
#     "predictor": ['SequenceClassificationPredictor',
#                   'FewshotSequenceClassificationPredictor', 
#                   'CptFewshotSequenceClassificationPredictor'],
# }

# if TYPE_CHECKING:
#     from .data import ClassificationDataset, DistillatoryClassificationDataset, FewshotSequenceClassificationDataset
#     from .evaluator import SequenceClassificationEvaluator, SequenceMultiLabelClassificationEvaluator
#     from .model import SequenceClassification, SequenceMultiLabelClassification, FewshotClassification, CPTClassification, DistillatoryBaseApplication
#     from .predictor import SequenceClassificationPredictor, FewshotSequenceClassificationPredictor, CptFewshotSequenceClassificationPredictor

# else:
#     import importlib
#     import os
#     import sys

#     class _LazyModule(_BaseLazyModule):
#         """
#         Module class that surfaces all objects but only performs associated imports when the objects are requested.
#         """

#         __file__ = globals()["__file__"]
#         __path__ = [os.path.dirname(__file__)]

#         def _get_module(self, module_name: str):
#             return importlib.import_module("." + module_name, self.__name__)

#     sys.modules[__name__] = _LazyModule(__name__, _import_structure)