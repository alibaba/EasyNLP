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
import tarfile
import time
from collections import defaultdict

import torch
import torch.nn as nn

from .arguments import is_torchx_available
from .global_vars import get_args
from .initializer import initialize_easynlp
from .io_utils import IO, OSSIO, TFOSSIO, DefaultIO, io, parse_oss_buckets
from .logger import init_logger

easynlp_default_path = os.path.join(os.environ["HOME"], ".easynlp")
EASYNLP_REMOTE_ROOT = "https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp"
EASYNLP_REMOTE_MODELZOO = "http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/"
EASYNLP_CACHE_ROOT = os.getenv("EASYNLP_CACHE_ROOT", easynlp_default_path)
EASYNLP_LOCAL_MODELZOO = os.path.join(EASYNLP_CACHE_ROOT, "modelzoo")
EASYNLP_LOCAL_DATAHUB = os.path.join(EASYNLP_CACHE_ROOT, "datahub")
EASYNLP_LOCAL_APPZOO = os.path.join(EASYNLP_CACHE_ROOT, "appzoo")

def copy_weights_for_same_module(copied_module, copying_module):
    if isinstance(copied_module, nn.Parameter):
        copying_module.data.copy_(copied_module.data)
        return
    tp = copied_module.named_parameters()
    for name, weights in copying_module.named_parameters():
        _, weights_ = next(tp)
        weights.data.copy_(weights_.data)


def get_dir_name(file_path):
    if io.isdir(file_path):
        return file_path
    else:
        return os.path.dirname(file_path)


def batch_dict_list_to_dict(lst):
    rst = defaultdict(list)
    for record in lst:
        for key, val in record.items():
            rst[key].append(val)
    return rst


def unbatch_dict_to_dict_list(lst_dict):
    rst = list()
    bsize = 1
    for key, val in lst_dict.items():
        bsize = len(val)
        break
    for b in range(bsize):
        tmp = dict()
        for key, val in lst_dict.items():
            tmp[key] = val[b]
        rst.append(tmp)
    return rst


def parse_row_by_schema(row: str, input_schema: str) -> dict:
    row_dict = dict()
    for schema, content in zip(input_schema.split(','),
                               row.strip('\n').split('\t')):
        col_name, col_type, col_length = schema.split(':')
        col_length = int(col_length)

        if col_type == 'str':
            row_dict[col_name] = content
        elif col_type == 'int':
            if col_length == 1:
                row_dict[col_name] = int(content)
            else:
                row_dict[col_name] = list(map(int, content.split(',')))
        elif col_type == 'float':
            if col_length == 1:
                row_dict[col_name] = float(content)
            else:
                row_dict[col_name] = list(map(float, content.split(',')))
        else:
            raise RuntimeError('Invalid schema: %s' % schema)
    return row_dict


def get_pretrain_model_path(pretrained_model_name_or_path,
                            disable_auto_download=False):
    if pretrained_model_name_or_path is None or  \
        pretrained_model_name_or_path.strip() == '':
        return None

    if pretrained_model_name_or_path.startswith('./') or \
        pretrained_model_name_or_path.startswith('../') or \
        pretrained_model_name_or_path.startswith('/') or \
        pretrained_model_name_or_path.startswith('oss://'):
        return pretrained_model_name_or_path

    # Use default $HOME/.easynlp_modelzoo as the modelzoo directory
    n_gpu = int(os.environ.get('EASYNLP_N_GPUS', '0'))
    modelzoo_base_dir = EASYNLP_LOCAL_MODELZOO
    is_master_node = (os.environ.get('EASYNLP_IS_MASTER',
                                     'true').lower() == 'true')

    if not io.exists(modelzoo_base_dir):
        io.makedirs(modelzoo_base_dir)
    assert io.isdir(modelzoo_base_dir
                    ), '%s is not a existing directory' % modelzoo_base_dir
    if not io.exists(modelzoo_base_dir + 'modelzoo_alibaba.json'):
        # Use the remote mapping file
        """with urllib.request.urlopen("http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/modelzoo_alibaba.json") as f:

        model_name_mapping = json.loads(f.read().decode('utf-8'))
        """
        while True:
            try:
                if os.path.exists('modelzoo_alibaba.json'):
                    break
                print('Trying downloading name_mapping.json')
                os.system(
                    'wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/modelzoo_alibaba.json'
                )
                print('Success')
            except Exception:
                time.sleep(2)

        with open('modelzoo_alibaba.json') as f:
            model_name_mapping = json.loads(f.read())
    else:
        with io.open(modelzoo_base_dir + 'modelzoo_alibaba.json') as f:
            model_name_mapping = json.load(f)
    if pretrained_model_name_or_path in model_name_mapping:
        pretrained_model_name = pretrained_model_name_or_path
        pretrained_model_name_or_path = model_name_mapping[
            pretrained_model_name_or_path]
        if 'oss://' in modelzoo_base_dir:
            # If the modelzoo is put on OSS buckets, do not need to download
            pretrained_model_name_or_path = os.path.join(
                modelzoo_base_dir, pretrained_model_name_or_path)
            assert io.exists(os.path.join(get_dir_name(pretrained_model_name_or_path), 'config.json')), \
                '%s not exists in OSS' % pretrained_model_name_or_path
        else:
            if not disable_auto_download:
                # Download the model tar file and untar the files (do once in master node while distributed training)
                remote_url = 'http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/' + \
                                pretrained_model_name_or_path
                local_tar_file_path = os.path.join(
                    modelzoo_base_dir, pretrained_model_name_or_path)
                if io.exists(local_tar_file_path) or io.isdir(
                        local_tar_file_path.replace('.tgz', '')):
                    print('`%s` already exists' % (local_tar_file_path))
                    return local_tar_file_path.replace('.tgz', '')
                else:
                    if is_master_node:
                        print('Downloading `%s` to %s' %
                              (pretrained_model_name, local_tar_file_path))
                        #print("Downloading `%s` to %s" % (remote_url, local_tar_file_path))
                        if not io.exists(get_dir_name(local_tar_file_path)):
                            io.makedirs(get_dir_name(local_tar_file_path))

                        os.system('wget ' + remote_url + ' -P ' +
                                  get_dir_name(local_tar_file_path))

                        try:
                            tar = tarfile.open(local_tar_file_path, 'r:gz')
                            pretrained_model_name_or_path = os.path.join(
                                modelzoo_base_dir, pretrained_model_name_or_path)
                            tar.extractall(
                                get_dir_name(pretrained_model_name_or_path))
                            tar.close()
                            os.system('rm -rf %s*' % local_tar_file_path)
                        except:
                            print('file %s not exists, deletion terminated.' % local_tar_file_path)
                            pass

                if n_gpu > 1:
                    torch.distributed.barrier()

            pretrained_model_name_or_path = os.path.join(
                EASYNLP_LOCAL_MODELZOO,
                model_name_mapping[pretrained_model_name].replace('.tgz', ''))
        return pretrained_model_name_or_path
    else:
        error_msg = "`%s` is not a existing pre-defined model name. Here're the list: \n" \
                    % pretrained_model_name_or_path
        for key in model_name_mapping.keys():
            error_msg += '\t' + key + '\n'
        raise RuntimeError(error_msg)


def parse_tf_config():
    """parse TF_CONFIG and return cluster, task info.

    Return
      cluster  a dict of cluster info
      task_type  string, if local mode, master will be returned
      task_index  int,  if local mode, 0 will be returned
    """
    tf_config_str = os.environ.get('TF_CONFIG', None)
    print(tf_config_str)
    if tf_config_str is None:
        return None, None, None
    else:
        tf_config = json.loads(tf_config_str)
        if 'cluster' not in tf_config or 'task' not in tf_config or \
                'type' not in tf_config['task'] or 'index' not in tf_config['task']:
            return None, None, None
        cluster = tf_config['cluster']
        task = tf_config['task']
        task_type = task['type']
        task_index = task['index']
        return cluster, task_type, task_index


def get_cnn_vocab(args):
    #TODO: a temp folder, need to clean up
    if os.path.exists(os.path.join(args.checkpoint_dir, 'vocab.txt')):
        return os.path.join(args.checkpoint_dir, 'vocab.txt')
    elif args.pretrained_model_name_or_path.endswith('en'):
        return os.path.join(
            os.path.join(os.environ['HOME'], '.easynlp', 'modelzoo'),
            get_pretrain_model_path('bert-small-uncased',
                                    disable_auto_download=True), 'vocab.txt')
    elif args.pretrained_model_name_or_path.endswith('zh'):
        return os.path.join(
            os.path.join(os.environ['HOME'], '.easynlp', 'modelzoo'),
            get_pretrain_model_path('bert-small-chinese',
                                    disable_auto_download=True), 'vocab.txt')
    else:
        raise ValueError('Language not supported')

