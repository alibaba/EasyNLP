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

import json
import os
import torch
from utils import io
from utils.logger import logger


def export_easytransfer_train_config(saved_path,
                                     vocab_dir,
                                     label_enumerate_values,
                                     sequence_length,
                                     model_name,
                                     extra_model_params):
    """ Save `train_config.json` for EasyTransfer AppZoo

        Args:
            saved_path (`str`) : the path of `train_config.json`
            vocab_dir (`str`) : the directory of `vocab.txt`
            label_enumerate_values (`list`) : The enumerate values of the label
            sequence_length (`int`) : Sequence Length while pre-processing
            model_name (`str`) : The model name of AppZoo, e.g. text_classify_bert
    """
    if isinstance(label_enumerate_values, list):
        num_label = len(label_enumerate_values)
        label_enumerate_values = ",".join(label_enumerate_values)
    else:
        label_enumerate_values = label_enumerate_values
        num_label = None

    if "oss://" in vocab_dir:
        pretrain_model_name_or_path = vocab_dir + "/model.ckpt"
    else:
        pretrain_model_name_or_path = os.path.abspath(vocab_dir) + "/model.ckpt"

    model_config_dict = dict()
    for key, val in extra_model_params.items():
        model_config_dict[key] = val

    model_config_dict["pretrain_model_name_or_path"] = pretrain_model_name_or_path
    model_config_dict["model_name"] = model_name
    model_config_dict["num_labels"] = num_label
    model_config_dict["dropout_rate"] = 0.0

    train_config_dict = {
        "_config_json": {
            "model_config": model_config_dict
        },
        "model_config": model_config_dict,
        "label_enumerate_values": label_enumerate_values,
        "sequence_length": sequence_length
    }
    with io.open(saved_path, "w") as f:
        f.write(json.dumps(train_config_dict, ensure_ascii=False, indent=4))


def export_pytorch_checkpoint_to_tf(model, ckpt_dir,
                                    bert_output_prefix="bert",
                                    appended_val_map=(),
                                    appended_tensors_to_transpose=()):
    """ Export PyTorch BERT model to TF Checkpoint

        Args:
            model (`nn.Module`) : The PyTorch model you want to save
            ckpt_dir (`str) : The directory of exporting checkpoint
            bert_output_prefix (`str`) : The prefix of BERT module, e.g. bert_pre_trained_model for EasyTransfer
            appended_val_map (`tuple`): A tuple of tuples, ( (PyTorch_var_name, Tensorflow_var_name), ...) )
            appended_tensors_to_transpose (`tuple`): A tuple of PyTorch tensor names you need to transpose
    """
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import re
        import numpy as np
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
    except ImportError:
        logger.info("Export a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
              "https://www.tensorflow.org/install/ for installation instructions.")
        raise RuntimeError

    def to_tf_var_name(name):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return name

    def create_tf_var(tensor, name, session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    var_map = appended_val_map + (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("/weight", "/kernel"))


    tensors_to_transpose = ("dense.weight",
                            "attention.self.query",
                            "attention.self.key",
                            "attention.self.value") + appended_tensors_to_transpose

    if not os.path.isdir(ckpt_dir):
        io.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    have_cls_predictions = False
    have_cls_seq_relationship = False
    for key in state_dict.keys():
        if key.startswith("cls.predictions"):
            have_cls_predictions = True
        if key.startswith("cls.seq_relationship"):
            have_cls_seq_relationship = True
    if not have_cls_predictions:
        state_dict["cls.predictions.output_bias"] = torch.zeros(model.config.vocab_size)
        state_dict["cls.predictions.transform.LayerNorm.beta"] = torch.zeros(model.config.hidden_size)
        state_dict["cls.predictions.transform.LayerNorm.gamma"] = torch.zeros(model.config.hidden_size)
        state_dict["cls.predictions.transform.dense.bias"] = torch.zeros(model.config.hidden_size)
        state_dict["cls.predictions.transform.dense.kernel"] = torch.zeros(
            (model.config.hidden_size, model.config.hidden_size))
    if not have_cls_seq_relationship:
        state_dict["cls.seq_relationship.output_weights"] = torch.zeros((2, model.config.hidden_size))
        state_dict["cls.seq_relationship.output_bias"] = torch.zeros(2)

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].cpu().numpy()
            if var_name.startswith("bert.") or var_name.startswith("cls."):
                prefix = bert_output_prefix + "/" if bert_output_prefix else ""
            else:
                prefix = ""
            tf_name = prefix + tf_name
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            # tf_weight = session.run(tf_var)
            # print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))
        create_tf_var(tensor=np.array(1), name="global_step", session=session)
        saver = tf.train.Saver(tf.trainable_variables())

        if "oss://" in ckpt_dir:
            saver.save(session, "model.ckpt")

            for fname in io.listdir("./"):
                if fname.startswith("model.ckpt"):
                    local_file = fname
                    oss_file = os.path.join(ckpt_dir, fname)
                    logger.info("uploading %s" % oss_file)
                    io.upload(local_file, oss_file)
        else:
            saver.save(session, os.path.join(ckpt_dir, "model.ckpt"))