# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and The HuggingFace Inc. team.
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


from abc import ABC
from typing import Any
from ..appzoo import SequenceClassificationPredictor

class Pipeline(ABC):
    """
    The Pipeline class is the class from which all pipelines inherit. 
    You can build pipelines quickly by taking 'Predictor' code.

    Example:
        
        SequenceClassificationPipeline(SequenceClassificationPredictor, Pipeline)
        SequenceLabelPipeline(SequenceLablingPredictor, Pipeline)

        Then get the output you want by overriding the __call__ function.
    """
    def __call__(self, inputs) -> dict:
        inputs = self.format_input(inputs)
        model_inputs = self.preprocess(inputs)
        model_outputs = self.predict(model_inputs)
        results = self.postprocess(model_outputs)
        return results
    
    def format_input(self, inputs):
        if type(inputs) != str and type(inputs) != list:
            raise RuntimeError("Input only supports strings or lists of strings")
        if type(inputs) == str:
            inputs = [inputs]
        return [{'first_sequence': input_sentence} for input_sentence in inputs]

class SequenceClassificationPipeline(SequenceClassificationPredictor, Pipeline):
    """
        You need to post-process the outputs of the __call__ to get the fields you need.
    """
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        results = super().__call__(*args, **kwds)
        if type(results) == dict:
            return [{'label': results['predictions']}]
        elif type(results) == list:
            return [{'label': res['predictions']} for res in results]
        else:
            raise NotImplementedError
    
    