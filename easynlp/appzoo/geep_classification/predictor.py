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
import numpy as np
from threading import Lock
import torch
from ...core.predictor import Predictor, get_model_predictor
from ...modelzoo import AutoTokenizer
from ...utils import io
from ...fewshot_learning.fewshot_predictor import PromptPredictor, CPTPredictor

class GEEPClassificationPredictor(Predictor):
    def __init__(self, model_dir, model_cls=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "oss://" in model_dir:
            local_dir = model_dir.split("/")[-1]
            local_dir = os.path.join("~/.cache", local_dir)
            os.makedirs(local_dir, exist_ok=True)
            io.copytree(model_dir, local_dir)
            model_dir = local_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.MUTEX = Lock()
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

    def preprocess(self, in_data):
        if not in_data:
            raise RuntimeError("Input data should not be None.")
        if not isinstance(in_data, list):
            in_data = [in_data]
        rst = {"id": [], "input_ids": [], "attention_mask": [], "token_type_ids": []}
        max_seq_length = -1
        for record in in_data:
            if "sequence_length" not in record:
                break
            max_seq_length = max(max_seq_length, record["sequence_length"])
        max_seq_length = self.sequence_length if (max_seq_length == -1) else max_seq_length
        # print("max_seq_length {}".format(max_seq_length))
        for record in in_data:
            text_a = record[self.first_sequence]
            text_b = record.get(self.second_sequence, None)
            try:
                self.MUTEX.acquire()
                feature = self.tokenizer(text_a,
                                         text_b,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=max_seq_length)
            finally:
                self.MUTEX.release()
            # print("len input ids {}".format(len(feature["input_ids"])))
            # print("len attention_mask {}".format(len(feature["attention_mask"])))
            # print("len token_type_ids {}".format(len(feature["token_type_ids"])))
            rst["id"].append(record.get("id", str(uuid.uuid4())))
            rst["input_ids"].append(feature["input_ids"])
            rst["attention_mask"].append(feature["attention_mask"])
            rst["token_type_ids"].append(feature["token_type_ids"])
        return rst

    def predict(self, in_data):
        return self.model_predictor.predict(in_data)

    def postprocess(self, result):
        probs = result["probabilities"]
        logits = result["logits"]
        predictions = np.argsort(-probs, axis=-1)
        new_results = list()
        for b, preds in enumerate(predictions):
            new_result = list()
            for pred in preds:
                new_result.append({
                    "pred": self.label_id_to_name[pred],
                    "prob": float(probs[b][pred]),
                    "logit": float(logits[b][pred])
                })
            new_results.append({
                "id": result["id"][b] if "id" in result else str(uuid.uuid4()),
                "output": new_result,
                "predictions": new_result[0]["pred"],
                "probabilities": ",".join([str(t) for t in result["probabilities"][b]]),
                "logits": ",".join([str(t) for t in result["logits"][b]])
            })
        if len(new_results) == 1:
            new_results = new_results[0]
        return new_results