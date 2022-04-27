# coding=utf-8
# Copyright (c) 2021 Alibaba PAI Team.
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

import torch
from datasets import load_dataset, load_metric
from easynlp.modelzoo import AutoTokenizer
from easynlp.appzoo.api import get_application_model
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import losses
from rapidformer import RapidformerEngine, get_args, Finetuner

class EasyNLPFintuner(Finetuner):
    def __init__(self,
                 engine,
                 ):
        super().__init__(engine=engine)

    def train_valid_test_datasets_provider(self):
        args = get_args()
        """Build train and validation dataset."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
            return outputs

        datasets = load_dataset(args.data_dir, args.data_name)

        # Apply the method we just defined to all the examples in all the splits of the dataset
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )
        tokenized_datasets.rename_column_("label", "labels")

        train_dataset = tokenized_datasets["train"]
        valid_dataset = tokenized_datasets['validation']
        test_dataset = tokenized_datasets['test']

        def collate_fn(examples):
            return tokenizer.pad(examples, padding="longest", return_tensors="pt")

        return train_dataset, valid_dataset, test_dataset, collate_fn

    def model_optimizer_lr_scheduler_provider(self):
        args = get_args()
        user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
        model = get_application_model(app_name=args.app_name,
                                      pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                      user_defined_parameters=user_defined_parameters)

        return model, None, None

    def run_forward_step(self, batch, model):
        label_ids = batch['labels']
        del batch['labels']
        forward_outputs = model(batch)
        logits = forward_outputs['logits']
        loss = losses.cross_entropy(logits, label_ids)
        return loss

    # after each epoch run metric on eval dataset
    def run_compute_metrics(self, model, eval_dataloader):
        args = get_args()
        model = model[0]
        metric = load_metric(args.data_dir, args.data_name)
        for step, batch in enumerate(eval_dataloader):
            label_ids = batch['labels']
            del batch['labels']
            with torch.no_grad():
                forward_outputs = model(batch)
            predictions = forward_outputs['predictions']
            metric.add_batch(
                predictions=self.gather(predictions),
                references=self.gather(label_ids),
            )

        eval_metric = metric.compute()
        return eval_metric

if __name__ == "__main__":
    engine = RapidformerEngine()
    trainer = EasyNLPFintuner(engine=engine)
    trainer.train()
