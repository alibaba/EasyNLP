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

import json
import os
import uuid

import torch
from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer
from ...utils import io
from ...modelzoo import BertTokenizer


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None, guid=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.guid = guid


class LabelingFeatures(object):
    """A single set of features of data for sequence labeling."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 all_tokens,
                 label_ids,
                 tok_to_orig_index,
                 seq_length=None,
                 guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.all_tokens = all_tokens
        self.seq_length = seq_length
        self.label_ids = label_ids
        self.tok_to_orig_index = tok_to_orig_index
        self.guid = guid


def bert_labeling_convert_example_to_feature(example, tokenizer, max_seq_length, label_map=None):
    """ Convert `InputExample` into `InputFeature` For sequence labeling task

        Args:
            example (`InputExample`): an input example
            tokenizer (`BertTokenizer`): BERT Tokenizer
            max_seq_length (`int`): Maximum sequence length while truncating
            label_map (`dict`): a map from label_value --> label_idx,
                                "regression" task if it is None else "classification"
        Returns:
            feature (`InputFeatures`): an input feature
    """

    content_tokens = example.text_a.split(" ")
    if example.label is not None:
        label_tags = example.label.split(" ")
    else:
        label_tags = None

    all_tokens = ["[CLS]"]
    all_labels = [""]
    tok_to_orig_index = [-100]
    for i, token in enumerate(content_tokens):
        sub_tokens = tokenizer.tokenize(token)
        if not sub_tokens:
            sub_tokens = ["[UNK]"]
        all_tokens.extend(sub_tokens)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        if label_tags is None:
            all_labels.extend(["" for _ in range(len(sub_tokens))])
        else:
            all_labels.extend([label_tags[i] for _ in range(len(sub_tokens))])
    all_tokens = all_tokens[:max_seq_length - 1]
    all_labels = all_labels[:max_seq_length - 1]
    all_tokens.append("[SEP]")
    all_labels.append("")
    tok_to_orig_index.append(-100)

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    label_ids = [label_map[label] if label else -100 for label in all_labels]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(-100)

    feature = LabelingFeatures(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_ids=label_ids,
                               all_tokens=all_tokens,
                               seq_length=max_seq_length,
                               tok_to_orig_index=tok_to_orig_index,
                               guid=example.guid)

    return feature


class SequenceLabelingPredictor(Predictor):

    def __init__(self, model_dir, model_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model_predictor = get_model_predictor(
            model_dir=model_dir,
            model_cls=model_cls,
            input_keys=[("input_ids", torch.LongTensor), ("attention_mask", torch.LongTensor),
                        ("token_type_ids", torch.LongTensor)],
            output_keys=["predictions", "probabilities", "logits"])
        self.label_path = os.path.join(model_dir, "label_mapping.json")
        with io.open(self.label_path) as f:
            self.label_mapping = json.load(f)
        self.label_id_to_name = {idx: name for name, idx in self.label_mapping.items()}
        self.first_sequence = kwargs.pop("first_sequence", "first_sequence")
        self.second_sequence = kwargs.pop("second_sequence", "second_sequence")
        self.sequence_length = kwargs.pop("sequence_length", 128)
        self.tokenized = kwargs.pop("tokenized", False)

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")

        if not isinstance(in_data, list):
            in_data = [in_data]

        rst = {
            "id": [],
            "input_ids": [],
            "attention_mask": [],
            "token_type_ids": [],
            "all_tokens": []
        }

        max_seq_length = -1
        for record in in_data:
            if not "sequence_length" in record:
                break
            max_seq_length = max(max_seq_length, record["sequence_length"])
        max_seq_length = self.sequence_length if (max_seq_length == -1) else max_seq_length

        for record in in_data:
            text_a = record[self.first_sequence] if self.tokenized else " ".join(
                record[self.first_sequence])
            example = InputExample(text_a=text_a, text_b=None, label=None)
            feature = bert_labeling_convert_example_to_feature(example, self.tokenizer,
                                                               max_seq_length)
            rst["id"].append(record.get("id", str(uuid.uuid4())))
            rst["input_ids"].append(feature.input_ids)
            rst["attention_mask"].append(feature.input_mask)
            rst["token_type_ids"].append(feature.segment_ids)
            rst["all_tokens"].append(feature.all_tokens)

        return rst

    def predict(self, in_data):
        return self.model_predictor.predict(in_data)

    def postprocess(self, result):

        def post_process_one_pred(raw_pred, tokens, label_mapping):
            tags = [label_mapping[t] for t in raw_pred]
            words = list()
            i = 0
            while True:
                if i >= len(tokens):
                    break
                if tags[i].startswith("B") and tokens[i] not in ["[CLS]", "[SEP]"]:
                    tag_type = tags[i].split("-")[-1]
                    st = i
                    i += 1
                    while i < len(tokens) and tags[i] == "I-" + tag_type and tokens[i] != "[SEP]":
                        i += 1
                    end = i
                    words.append({
                        "word": "".join(tokens[st:end]),
                        "tag": tag_type,
                        "start": st - 1,
                        "end": end - 1
                    })
                else:
                    i += 1
            return words, tags

        new_results = list()
        for b, (predictions, tokens) in enumerate(zip(result["predictions"], result["all_tokens"])):
            words, tags = post_process_one_pred(predictions, tokens, self.label_id_to_name)
            new_results.append({
                "id": result["id"][b] if "id" in result else str(uuid.uuid4()),
                "output": words,
                "predictions": " ".join(tags),
            })
        if len(new_results) == 1:
            new_results = new_results[0]
        return new_results
