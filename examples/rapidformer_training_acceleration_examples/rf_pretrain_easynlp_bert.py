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
from easynlp.appzoo.api import get_application_model
from easynlp.utils.global_vars import parse_user_defined_parameters
from rapidformer import mpu, RapidformerEngine, get_args, PreTrainer, build_pretrain_huggingface_bert_datasets

class EasyNLPRoBertaPreTrainer(PreTrainer):

    def __init__(self,engine):
        super().__init__(engine=engine)

    def train_valid_test_datasets_provider(self, train_val_test_num_samples):
        args = get_args()

        train_ds, valid_ds, test_ds = build_pretrain_huggingface_bert_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            max_seq_length=args.seq_length,
            masked_lm_prob=args.mask_prob,
            short_seq_prob=args.short_seq_prob,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            binary_head=True)

        return train_ds, valid_ds, test_ds

    def model_optimizer_lr_scheduler_provider(self):
        args = get_args()
        user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
        model = get_application_model(app_name=args.app_name,
                                      pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                      user_defined_parameters=user_defined_parameters)

        return model.backbone, None, None

    def run_forward_step(self, data_iterator, model):
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'next_sentence_label']
        datatype = torch.int64

        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)
        input_ids = data_b['input_ids'].long()
        attention_mask = data_b['attention_mask'].long()
        token_type_ids = data_b['token_type_ids'].long()
        labels = data_b['labels'].long()
        output_tensor = model(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids, labels=labels)

        return output_tensor['loss']

if __name__ == "__main__":
    engine = RapidformerEngine()
    trainer = EasyNLPRoBertaPreTrainer(engine=engine)
    trainer.train()
