# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team and Alibaba PAI team.
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
import urllib
import logging
import json
import tarfile
from typing import Any, List, Optional
from ..appzoo import TextImageGeneration, CLIPGPTImageTextGeneration, \
        SequenceClassification, TextMatch, SequenceLabeling, MachineReadingComprehension, LatentDiffusion, StableDiffusion
from ..utils.io_utils import io
from .implementation import Pipeline, TextImageGenerationPipeline, ImageTextGenerationPipeline, \
        SequenceClassificationPipeline, TextImageGenerationPipeline, \
        TextMatchPipeline, SequenceLabelingPipeline, MachineReadingComprehensionPipeline, LatentDiffusionPipeline

from ..utils import EASYNLP_CACHE_ROOT, EASYNLP_REMOTE_MODELZOO, EASYNLP_LOCAL_APPZOO


logger = logging.getLogger(__name__)
# Some tasks with different names but the same processing flow are mapped here.
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "question-answer": "text_match",
}

SUPPORTED_TASKS = {
    'text_classify': {
        'impl': SequenceClassificationPipeline,
        'model_cls': SequenceClassification,
        'default': 'bert-base-sst', 
    }, 
    'text_match': {
        'impl': TextMatchPipeline,
        'model_cls': TextMatch,
        'default': 'bert-small-qnli', 
    },
    'sequence_labeling': {
        'impl': SequenceLabelingPipeline,
        'model_cls': SequenceLabeling,
        'default': 'chinese-roberta-basener',
    },
    'text2image_generation': {
        'impl': TextImageGenerationPipeline,
        'model_cls': TextImageGeneration,
        'default': 'artist-base-zh',
    },
    'chinese-ldm-general': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-general',
    },
    'stable-diffusion-general': {
        'impl': LatentDiffusionPipeline,
        'model_cls': StableDiffusion,
        'default': 'stable-diffusion-general',
    },
    'chinese-ldm-fashion': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-fashion',
    },
    'chinese-ldm-art': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-art',
    },
    'chinese-ldm-poem': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-poem',
    },
    'chinese-ldm-anime': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-anime',
    },
    'chinese-ldm-pet': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-pet',
    },
    'chinese-ldm-food': {
        'impl': LatentDiffusionPipeline,
        'model_cls': LatentDiffusion,
        'default': 'chinese-ldm-food',
    },
    'image2text_generation': {
        'impl': ImageTextGenerationPipeline,
        'model_cls': CLIPGPTImageTextGeneration,
        'default': 'clip-gpt-i2t-base-zh',
    },
    'machine_reading_comprehension': {
        'impl': MachineReadingComprehensionPipeline,
        'model_cls': MachineReadingComprehension,
        'default': 'macbert-base-rczh',
    }
}

def pipeline(
    task_or_model: str = None,
    model_path: str = None,
    pipeline_class: Optional[Any] = None,
    **kwargs 
) -> Pipeline:
    """
    Utility factory method to build a [`Pipeline`].

    Args:

    """
    if task_or_model is None:
        raise RuntimeError("You must specify a task or model to initialize the pipeline,  \
                            invoking 'get_supported_tasks()' to get task list, \
                            invoking 'get_easynlp_model_list()' to get easynlp task model list.")
    support_task = get_supported_tasks()
    app_name = None
    app_setting = None
    if task_or_model in support_task:
        app_name = task_or_model
        app_setting = SUPPORTED_TASKS[app_name]
        if model_path is None:
            model_path = get_app_model_path(app_setting['default'])
    elif task_or_model in get_supported_app_model(False):
        logger.info("You are using easynlp model, the parameter named 'model_path' is ignored.")
        model_name = task_or_model
        model_path = get_app_model_path(model_name)
        app_name = get_remote_app_model_mapping()[model_name]['app_name']
        app_setting = SUPPORTED_TASKS[app_name]
    else:
        raise RuntimeError("You must specify a app and corresponding model to initialize the pipeline,  \
                            invoking 'get_supported_tasks()' to get task list, \
                            invoking 'get_easynlp_model_list()' to get easynlp task model list. \
                            For example: pipe_ = pipeline('text_classify', './classify_model/bert-base-sst2')")
    assert app_setting is not None
    pipeline_class = app_setting['impl']
    model_cls = app_setting['model_cls']
    return pipeline_class(model_path, model_cls, kwargs)


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    supported_tasks = list(SUPPORTED_TASKS.keys()) + list(TASK_ALIASES.keys())
    supported_tasks.sort()
    return supported_tasks

def get_remote_app_model_mapping() -> dict:
    """
    Returns a dict of supported model. It's a tool function for get_supported_app_model().
    """
    remote_base = EASYNLP_REMOTE_MODELZOO
    cache_root = EASYNLP_CACHE_ROOT
    remote_file_path = os.path.join(remote_base, 'appzoo_config.json')
    local_file_path = os.path.join(cache_root, "appzoo_config.json")
    try:
        if not os.path.exists(cache_root):
            os.makedirs(cache_root)
        urllib.request.urlretrieve(remote_file_path, local_file_path)
    except:
        if os.path.exists(local_file_path):
            logging.info("Unable to get the latest version of 'easynlp_trained_model_config.json', \
                            use the old version.")
        else:
            raise RuntimeError
    with io.open(local_file_path) as f:
        easynlp_model_mapping = json.loads(f.read())
    return easynlp_model_mapping

def get_supported_app_model(sort=True) -> dict:
    """
    Returns a dict of supported model sorted by model's app_name when 'srot' flag is True.
    """
    easynlp_model_mapping = get_remote_app_model_mapping()
    if sort == False:
        return list(easynlp_model_mapping)
    model_dict = dict()
    for model in list(easynlp_model_mapping):
        model_app_name = easynlp_model_mapping[model]["app_name"]
        if model_app_name not in model_dict:
            # supported_tasks.append(model_app_name)
            model_dict[model_app_name] = [model]
        else:
            model_dict[model_app_name].append(model)
    return model_dict

def get_app_model_path(model_name, disable_auto_download=False):
    """
    Download the remote app model to local machine and return the absolute path.
    The function returns the absolute path if the model already exists locally.
    """
    appzoo_base_dir = EASYNLP_LOCAL_APPZOO
    remote_modelzoo_url = EASYNLP_REMOTE_MODELZOO
    if not io.exists(appzoo_base_dir):
        io.makedirs(appzoo_base_dir)
    easynlp_app_mapping = get_remote_app_model_mapping()
    if model_name in easynlp_app_mapping:
        app_model_name = model_name
        remote_app_model_path = easynlp_app_mapping[
            app_model_name]['model_path']
        if 'oss://' in appzoo_base_dir:
            # If the modelzoo is put on OSS buckets, do not need to download
            app_local_path = os.path.join(
                appzoo_base_dir, remote_app_model_path)
            assert io.exists(os.path.join(os.path.dirname(app_local_path), 'config.json')), \
                '%s not exists in OSS' % app_local_path
        else:
            if not disable_auto_download:
                # Download the model tar file and untar the files (do once in master node while distributed training)
                remote_url = os.path.join(remote_modelzoo_url, remote_app_model_path)
                local_tar_file_path = os.path.join(
                    appzoo_base_dir, remote_app_model_path.split('/')[1])
                if io.isdir(
                        local_tar_file_path.replace('.tgz', '')):
                    logger.info('`%s` already exists' % (local_tar_file_path))
                    return local_tar_file_path.replace('.tgz', '')
                else:
                    if not io.exists(local_tar_file_path):
                        logger.info('Downloading `%s` to %s' %
                              (appzoo_base_dir, local_tar_file_path))
                        if not io.exists(os.path.dirname(local_tar_file_path)):
                            io.makedirs(os.path.dirname(local_tar_file_path))
                        os.system('wget ' + remote_url + ' -P ' +
                                  os.path.dirname(local_tar_file_path))
                    tar = tarfile.open(local_tar_file_path, 'r:gz')
                    local_app_model_path = local_tar_file_path.replace('.tgz', '')
                    tar.extractall(os.path.dirname(local_tar_file_path))
                    tar.close()
                    os.remove(local_tar_file_path)
        return local_app_model_path
    else:
        error_msg = "`%s` is not a existing pre-defined model name. Here're the list: \n" \
                    % model_name
        for key in easynlp_app_mapping.keys():
            error_msg += '\t' + key + '\n'
        raise RuntimeError(error_msg)
