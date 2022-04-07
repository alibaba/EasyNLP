# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team and The Alibaba PAI team.
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
import traceback

import torch
from torch import nn

from easynlp.appzoo.application import Application
from easynlp.modelzoo import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from easynlp.utils import io
from easynlp.utils.logger import logger
from easynlp.utils.losses import cross_entropy


class FewshotClassification(Application):
    """An application class for supporting fewshot learning (PET and P-tuning)."""
    def __init__(self,
                 pretrained_model_name_or_path=None,
                 user_defined_parameters=None,
                 **kwargs):
        super(FewshotClassification, self).__init__()
        if kwargs.get('from_config'):
            self.config = kwargs.get('from_config')
            self.backbone = AutoModelForMaskedLM.from_config(self.config)
        # for pretrained model, initialize from the pretrained model
        else:
            self.config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path)
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path)

        # if p-tuning, model's embeddings should be changed
        try:
            self.user_defined_parameters_dict = user_defined_parameters.get(
                'app_parameters')
        except KeyError:
            traceback.print_exc()
            exit(-1)
        pattern = self.user_defined_parameters_dict.get('pattern')
        assert pattern is not None, 'You must define the pattern for PET learning'
        pattern_list = pattern.split(',')
        cnt = 0
        for i in range(len(pattern_list)):
            if pattern_list[i] == '<pseudo>':
                pattern_list[i] = '<pseudo-%d>' % cnt
                cnt += 1
        if cnt > 0:
            self.tokenizer.add_tokens(['<pseudo-%d>' % i for i in range(cnt)])
        self.backbone.resize_token_embeddings(len(self.tokenizer))
        print('embedding size: %d' % len(self.tokenizer))
        self.config.vocab_size = len(self.tokenizer)

    def forward(self, inputs):
        if 'mask_span_indices' in inputs:
            inputs.pop('mask_span_indices')
        outputs = self.backbone(**inputs)
        return {'logits': outputs.logits}

    def compute_loss(self, forward_outputs, label_ids):
        prediction_scores = forward_outputs['logits']
        masked_lm_loss = cross_entropy(
            prediction_scores.view(-1, self.config.vocab_size),
            label_ids.view(-1))
        return {'loss': masked_lm_loss}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Instantiate model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            from_config=config,
            **kwargs)
        state_dict = None
        weights_path = os.path.join(pretrained_model_name_or_path,
                                    'pytorch_model.bin')
        if not io.exists(weights_path):
            return model
        with io.open(weights_path, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        logger.info('Loading model...')
        load(model, prefix=start_prefix)
        logger.info('Load finished!')
        if len(missing_keys) > 0:
            logger.info(
                'Weights of {} not initialized from pretrained model: {}'.
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info(
                'Weights from pretrained model not used in {}: {}'.format(
                    model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    model.__class__.__name__, '\n\t'.join(error_msgs)))
        return model


class CPTClassification(FewshotClassification):
    """An application class for supporting CPT fewshot learning."""
    def __init__(self,
                 pretrained_model_name_or_path=None,
                 user_defined_parameters=None,
                 **kwargs):
        super(CPTClassification,
              self).__init__(pretrained_model_name_or_path,
                             user_defined_parameters, **kwargs)

        circle_loss_config = self.user_defined_parameters_dict.get(
            'circle_loss_config')
        if circle_loss_config:
            circle_loss_config = json.loads(circle_loss_config)
        else:
            circle_loss_config = dict()
        self.loss_fcn = CircleLoss(**circle_loss_config)

    def forward(self, inputs, do_mlm=False):
        # currently CPT only supports models that share the structures with bert and hfl/chinese-roberta-wwm-ext
        if 'mask_span_indices' in inputs:
            inputs.pop('mask_span_indices')
        if 'label_ids' in inputs:
            inputs.pop('label_ids')
        if do_mlm:
            outputs = self.backbone(**inputs)
            return {'logits': outputs.logits}
        else:
            x = self.backbone.bert(**inputs)[0]
            outputs = self.backbone.cls.predictions.transform(x)
            return {'features': outputs}

    def compute_loss(self, forward_outputs, label_ids):
        features = forward_outputs['features'][label_ids > 0]
        features = nn.functional.normalize(features)
        labels = label_ids[label_ids > 0]
        loss = self.loss_fcn(features, labels)
        return {'loss': loss}


class CircleLoss(nn.Module):
    def __init__(self,
                 margin: float = 0.4,
                 gamma: float = 64,
                 k: float = 1,
                 distance_function='cos') -> None:
        super(CircleLoss, self).__init__()
        self.m = margin
        self.gamma = gamma
        self.k = k
        self.soft_plus = nn.Softplus()
        if distance_function == 'cos':
            self.dist_fcn = lambda X: X @ X.transpose(1, 0)
        else:
            raise NotImplementedError

    def forward(self, features, labels):
        sim = self.dist_fcn(features).view(-1)
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        pos = mask.triu(diagonal=1).view(-1)
        neg = mask.logical_not().triu(diagonal=1).view(-1)
        sp = sim[pos]
        sn = sim[neg]
        ap = (1 / self.k) * torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma
        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss
