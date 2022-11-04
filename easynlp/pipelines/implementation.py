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
from ..appzoo import CLIPGPTImageTextGenerationPredictor,\
                    TextImageGenerationPredictor, LatentDiffusionPredictor, \
                    SequenceClassificationPredictor, \
                    TextMatchPredictor, SequenceLabelingPredictor, \
                    MachineReadingComprehensionPredictor

class Pipeline(ABC):
    """
    The Pipeline class is the class from which all pipelines inherit. 
    You can build pipelines quickly by taking 'Predictor' code.

    Example:
        
        SequenceClassificationPipeline(SequenceClassificationPredictor, Pipeline)
        SequenceLabelPipeline(SequenceLablingPredictor, Pipeline)

    Then get the output you want by overriding the __call__ function.
    """
    def format_input(self, inputs):
        """
        Preprocess single sentence data.
        """
        if type(inputs) != str and type(inputs) != list:
            raise RuntimeError("Input only supports strings or lists of strings")
        if type(inputs) == str:
            inputs = [inputs]
        return [{'first_sequence': input_sentence} for input_sentence in inputs]

    def __call__(self, inputs) -> dict:
        inputs = self.format_input(inputs)
        model_inputs = self.preprocess(inputs)
        model_outputs = self.predict(model_inputs)
        results = self.postprocess(model_outputs)
        return results

class ImageTextGenerationPipeline(CLIPGPTImageTextGenerationPredictor, Pipeline):

    def format_input(self, inputs):
        """
        Preprocess single sentence data.
        """
        if type(inputs) != str and type(inputs) != list:
            raise RuntimeError("Input only supports string or lists of string")
        if type(inputs) == str:
            inputs = [inputs]

        return [{'idx': idx, \
                'first_sequence': inputs[idx]} for idx in range(len(inputs))]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        You need to post-process the outputs of the __call__ to get the fields you need.
        """
        results = super().__call__(*args, **kwds)
        return [{'gen_text': res['gen_text']} for res in results]

class TextImageGenerationPipeline(TextImageGenerationPredictor, Pipeline):

    def format_input(self, inputs):
        """
        Preprocess single sentence data.
        """
        if type(inputs) != str and type(inputs) != list:
            raise RuntimeError("Input only supports string or lists of string")
        if type(inputs) == str:
            inputs = [inputs]

        return [{'idx': idx, \
                'first_sequence': inputs[idx]} for idx in range(len(inputs))]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        You need to post-process the outputs of the __call__ to get the fields you need.
        """
        results = super().__call__(*args, **kwds)
        return [{'gen_imgbase64': res['gen_imgbase64']} for res in results]

class LatentDiffusionPipeline(LatentDiffusionPredictor, Pipeline):

    def format_input(self, inputs):
        """
        Preprocess single sentence data.
        """
        if type(inputs) != str and type(inputs) != list:
            raise RuntimeError("Input only supports string or lists of string")
        if type(inputs) == str:
            inputs = [inputs]
        return [{'idx': idx, \
                'text': inputs[idx]} for idx in range(len(inputs))]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        You need to post-process the outputs of the __call__ to get the fields you need.
        """
        results = super().__call__(*args, **kwds)
        return results

class SequenceClassificationPipeline(SequenceClassificationPredictor, Pipeline):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        You need to post-process the outputs of the __call__ to get the fields you need.
        """
        results = super().__call__(*args, **kwds)
        if type(results) == dict:
            return [{'label': results['predictions']}]
        elif type(results) == list:
            return [{'label': res['predictions']} for res in results]
        else:
            raise NotImplementedError

class TextMatchPipeline(TextMatchPredictor, Pipeline):
    """
    This is a implement of TextMatch pipeline. 
    Input format: 
        [sent1, sent2] or [[sent1, sent2], [sent1, sent2]]
    """
    def format_input(self, inputs):
        """
        Preprocess twin sentence data.
        """
        if type(inputs) != list:
            raise RuntimeError("'TextMatchPipeline' only supports lists! \
                    Every data instance contains two fields of sentence. \
                    For example: [sent1, sent2] or [[sent1, sent2], [sent1, sent2]]")
        if len(inputs) == 2:
            inputs = [inputs]
        assert len(inputs[0]) == 2
        return [{'first_sequence': input_sentence_pair[0],
                'second_sequence': input_sentence_pair[1]} for input_sentence_pair in inputs]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        results = super().__call__(*args, **kwds)
        if type(results) == dict:
            return [{'label': results['predictions']}]
        elif type(results) == list:
            return [{'label': res['predictions']} for res in results]
        else:
            raise NotImplementedError

class SequenceLabelingPipeline(SequenceLabelingPredictor, Pipeline):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        results = super().__call__(*args, **kwds)
        def delete_useless_key(_dict: dict, key_name: str) -> dict:
            _dict.pop(key_name)
            return _dict
        if type(results) == dict:
            return delete_useless_key(results, 'id')
        elif type(results) == list:
            return [delete_useless_key(res, 'id') for res in results]
        else:
            raise NotImplementedError

class MachineReadingComprehensionPipeline(MachineReadingComprehensionPredictor, Pipeline):

    """
    This is a implement of MachineReadingComprehension pipeline. 
    Input format: 
        {"context": context, "query": query, "answer": answer, "id": id} 
      or 
        [{"context": context1, "query": query1, "answer": answer1, "id": id1}, 
         {"context": context2, "query": query2, "answer": answer2, "id": id2}]
    """
    def format_input(self, inputs):
        if type(inputs) != dict and type(inputs) != list:
            raise RuntimeError("Input only supports strings or lists of strings")
        if type(inputs) == dict:
            inputs = [inputs]
        return inputs
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        You need to post-process the outputs of the __call__ to get the fields you need.
        """
        results = super().__call__(*args, **kwds)
        if type(results) == dict:
            return [results]
        elif type(results) == list:
            return [{"unique_id": res["unique_id"],
                     "best_answer": res["best_answer"],
                     "gold_answer": res["gold_answer"],
                     "query": res["query"],
                     "context": res["context"]} for res in results]
        else:
            raise NotImplementedError
