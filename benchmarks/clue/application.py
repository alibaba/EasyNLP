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

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from easynlp.appzoo.application import Application
from easynlp.modelzoo import AutoConfig, AutoModel
from easynlp.utils import losses
from easynlp.utils.logger import logger
from utils import CLUEDataset


class CLUEApp(Application):

    def __init__(self,
            args,
            task_name: str,
            app_name: str,
            pretrained_model_name_or_path: str,
            user_defined_parameters: dict,
            is_training: bool = False,
            num_labels: int = 2,
            **kwargs
    ):
        super().__init__()
        # obtain function values
        self.args = args
        self.task_name = task_name
        self.app_name = app_name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.user_defined_parameters = user_defined_parameters
        self.is_training = is_training
        logger.info("Training stage: {}".format(self.is_training))
        # for pretrained model, initialize from the pretrained model
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path)
        # adding configure to PLM config file
        self.config.num_labels = num_labels
        self.config.task_specific_params = {
            'task_name': self.task_name,
            'app_name': self.app_name,
        }
        if self.config.task_specific_params['app_name'] in ['text_classify', 'text_match']:
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        if self.is_training:
            self.init_weights()
        else:
            self.load_state_dict(torch.load(os.path.join(self.args.checkpoint_dir, 'pytorch_model.bin')))

    def init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        pooler_output = self.dropout(pooler_output)
        if self.config.task_specific_params['app_name'] in ['text_classify', 'text_match']:
            logits = self.classifier(pooler_output)
        return {
            'hidden': hidden_states,
            'logits': logits,
            'predictions': torch.argmax(logits, dim=-1),
            'probabilities': torch.softmax(logits, dim=-1)
        }

    def compute_loss(self, forward_outputs, label_ids, **kwargs):
        logits = forward_outputs['logits']
        return {'loss': losses.cross_entropy(logits, label_ids)}


class CLUEPredictor:
    def __init__(self,
            args,
            model,
            task_name: str,
            datatset: CLUEDataset,
            id2label: dict
        ):
        self.args = args
        self.model = model
        self.task_name = task_name
        self.dataset = datatset
        self.id2label = id2label
        if self.args.use_torchacc:
            import torchacc.torch_xla.core.xla_model as xm
            self._device = xm.xla_device()
            xm.set_replication(self._device, [self._device])
        self.set_data_loader(self.dataset, self.args)
        self.pred_output_dir = "./tmp/predict/clue/{}".format(task_name)
        if not os.path.exists(self.pred_output_dir):
            os.makedirs(self.pred_output_dir)

    def set_data_loader(self, dataset, args):

        if args.read_odps:
            data_sampler = None
        else:
            if self.args.use_torchacc:
                import torchacc.torch_xla.core.xla_model as xm
                if xm.xrt_world_size() > 1:
                    data_sampler = torch.utils.data.distributed.DistributedSampler(
                        dataset,
                        num_replicas=xm.xrt_world_size(),
                        rank=xm.get_ordinal(),
                        shuffle=True)
                else:
                    data_sampler = None
            elif args.n_gpu <= 1:
                data_sampler = RandomSampler(dataset)
            else:
                data_sampler = DistributedSampler(dataset)

        self.args.pred_batch_size = self.args.micro_batch_size * max(1, self.args.n_gpu)

        self._data_loader = DataLoader(dataset,
                                        sampler=data_sampler,
                                        batch_size=args.pred_batch_size,
                                        collate_fn=dataset.batch_fn,
                                        num_workers=self.args.data_threads)
        if self.args.use_torchacc:
            import torchacc.torch_xla.distributed.parallel_loader as pl
            self._data_loader = pl.MpDeviceLoader(self._data_loader,
                                                   self._device)


    def predict(self):

        logger.info("******** Running prediction ********")
        logger.info("  Num examples = %d", len(self.dataset))
        logger.info("  Batch size = %d", self.args.pred_batch_size)
        nb_pred_steps = 0
        preds = None
        for step, batch in enumerate(tqdm(self._data_loader)):
            self.model.eval()
            if not self.args.use_torchacc:
                batch = {
                    key: val.to(self.args.local_rank) if isinstance(
                        val, torch.Tensor) else val
                    for key, val in batch.items() if key != 'id'
                }
                # print('batch=', batch)
            with torch.no_grad():
                '''
                {
                    'hidden': hidden_states,
                    'logits': logits,
                    'predictions': torch.argmax(logits, dim=-1),
                    'probabilities': torch.softmax(logits, dim=-1)
                }
                '''
                outputs = self.model(batch)
                logits = outputs['logits']
            nb_pred_steps += 1
            if preds is None:
                if self.task_name == 'copa':
                    preds = logits.softmax(-1).detach().cpu().numpy()
                else:
                    preds = logits.detach().cpu().numpy()
            else:
                if self.task_name == 'copa':
                    preds = np.append(preds, logits.softmax(-1).detach().cpu().numpy(), axis=0)
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        print(' ')
        predict_label = np.argmax(preds, axis=1)

        if self.task_name == 'copa':
            predict_label = []
            pred_logits = preds[:, 1]
            i = 0
            while (i < len(pred_logits) - 1):
                if pred_logits[i] >= pred_logits[i + 1]:
                    predict_label.append(0)
                else:
                    predict_label.append(1)
                i += 2
        output_submit_file = os.path.join(self.pred_output_dir, "test_prediction.json")
        # save the predicted result
        with open(output_submit_file, "w") as writer:
            for i, pred in enumerate(predict_label):
                json_d = {}
                json_d['id'] = i
                json_d['label'] = str(self.id2label[pred])
                writer.write(json.dumps(json_d) + '\n')
