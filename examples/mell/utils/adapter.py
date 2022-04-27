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

import os

import torch

from utils import io
from utils.logger import logger


def load_bert_tf_checkpoint_weights(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model

        Args:
            saved_path (`str`) : the path of `train_config.json`
            vocab_dir (`str`) : the directory of `vocab.txt`
            label_enumerate_values (`list`) : The enumerate values of the label
            sequence_length (`int`) : Sequence Length while pre-processing
            model_name (`str`) : The model name of AppZoo, e.g. text_classify_bert
        Returns:
            model (`torch.nn.Module`) : A defined PyTorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.info("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise RuntimeError
    tf_path = io.abspath(tf_checkpoint_path)

    if "oss://" in tf_path:
        tmp_tf_path = "./easydistill_tmp_" + tf_path.split("/")[-1]
        ckpt_dir = os.path.dirname(tf_checkpoint_path)
        filenames = io.listdir(ckpt_dir)
        for fname in filenames:
            if fname.startswith(tf_path.split("/")[-1]):
                local_path = tmp_tf_path + fname.replace(tf_path.split("/")[-1], "")
                oss_path = os.path.join(ckpt_dir, fname)
                logger.info("Download %s" % oss_path)
                io.download(oss_path, local_path)
        tf_path = tmp_tf_path

    logger.info("Converting TensorFlow checkpoint to PyTorch")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        if "Adam" in name or "beta1_power" in name or "beta2_power" in name:
            continue
        if "global_step" in name:
            continue

        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    valid_count = 0
    for name, array in zip(names, arrays):
        name = name.split('/')
        if name[0] in ["bert_pre_trained_model", "roberta_pre_trained_model"]:
            name = name[1:]
        if name[0] == "text_match_bert_two_tower":
            name = name[2:]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step", "Adam", "Adam_1", "beta1_power", "beta2_power"] for n in name):
            # print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == "cls":
                break
            if l[0] == 'app':
                continue

            try:
                if l[0] == 'kernel' or l[0] == 'gamma':
                    pointer = getattr(pointer, 'weight')
                elif l[0] == l[0] == "bias" or l[0] == 'beta':
                    pointer = getattr(pointer, 'bias')
                elif l[0] == 'squad':
                    pointer = getattr(pointer, 'classifier')
                elif l[0] == 'ez_dense':
                    pointer =  getattr(pointer, 'classifier')
                elif l[0] == 'output_bias':
                    try:
                        pointer = getattr(pointer, 'bias')
                    except:
                        pointer = getattr(pointer.classifier, 'bias')
                elif l[0] == 'output_weights':
                    try:
                        pointer = getattr(pointer, 'weight')
                    except:
                        pointer = getattr(pointer.classifier, 'weight')
                else:
                    pointer = getattr(pointer, l[0])
            except AttributeError:
                logger.info("Skipping {}".format("/".join(name)))
                continue

            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if isinstance(pointer, type(model)):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except Exception as e:
            e.args += (pointer.shape, array.shape)
            continue
        # print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
        valid_count += 1
    logger.info("Convert finished!".format(tf_path))
    return model


def load_bert_mtl_tf_checkpoint_weights(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model

            Args:
                saved_path (`str`) : the path of `train_config.json`
                vocab_dir (`str`) : the directory of `vocab.txt`
                label_enumerate_values (`list`) : The enumerate values of the label
                sequence_length (`int`) : Sequence Length while pre-processing
                model_name (`str`) : The model name of AppZoo, e.g. text_classify_bert
            Returns:
                model (`torch.nn.Module`) : A defined PyTorch model
        """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.info("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
                    "https://www.tensorflow.org/install/ for installation instructions.")
        raise RuntimeError
    tf_path = io.abspath(tf_checkpoint_path)

    if "oss://" in tf_path:
        tmp_tf_path = "./easydistill_tmp_" + tf_path.split("/")[-1]
        ckpt_dir = os.path.dirname(tf_checkpoint_path)
        filenames = io.listdir(ckpt_dir)
        for fname in filenames:
            if fname.startswith(tf_path.split("/")[-1]):
                local_path = tmp_tf_path + fname.replace(tf_path.split("/")[-1], "")
                oss_path = os.path.join(ckpt_dir, fname)
                logger.info("Download %s" % oss_path)
                io.download(oss_path, local_path)
        tf_path = tmp_tf_path

    logger.info("Converting TensorFlow checkpoint to PyTorch")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        if "Adam" in name or "beta1_power" in name or "beta2_power" in name:
            continue
        if "global_step" in name:
            continue

        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    valid_count = 0
    for name, array in zip(names, arrays):
        name = name.split('/')
        if name[0] == "bert_pre_trained_model":
            name = name[1:]
        if name[0] == "text_match_bert_two_tower":
            name = name[2:]
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step", "Adam", "Adam_1", "beta1_power", "beta2_power"] for n in name):
            # print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == "cls":
                break
            if l[0] == 'app':
                continue
            if l[0] == 'output' and name[-1].startswith("multi_task"):
                continue

            try:
                if l[0] == 'kernel' or l[0] == 'gamma':
                    pointer = getattr(pointer, 'weight')
                elif l[0] == "bias" or l[0] == 'beta':
                    pointer = getattr(pointer, 'bias')
                elif l[0].startswith("multi_task"):
                    pointer = getattr(pointer.output, l[0])
                elif l[0] == 'squad':
                    pointer = getattr(pointer, 'classifier')
                elif l[0] == 'ez_dense':
                    pointer = getattr(pointer, 'classifier')
                elif l[0] == 'output_bias':
                    try:
                        pointer = getattr(pointer, 'bias')
                    except:
                        pointer = getattr(pointer.classifier, 'bias')
                elif l[0] == 'output_weights':
                    try:
                        pointer = getattr(pointer, 'weight')
                    except:
                        pointer = getattr(pointer.classifier, 'weight')
                else:
                    pointer = getattr(pointer, l[0])
            except AttributeError:
                logger.info("Skipping {}".format("/".join(name)))
                continue

            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if isinstance(pointer, type(model)):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except Exception as e:
            e.args += (pointer.shape, array.shape)
            continue
        # print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
        valid_count += 1
    logger.info("Convert finished!".format(tf_path))
    return model