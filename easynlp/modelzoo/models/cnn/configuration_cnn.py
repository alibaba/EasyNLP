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

from __future__ import absolute_import, division, print_function, unicode_literals
from ...configuration_utils import PretrainedConfig
from ..bert import BertConfig


CNN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "alibaba-pai/textcnn-en": "https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/textcnn-en/config.json",
}

# Currently not used, but will be used if we modify the from_pretrained method
class TextCNNConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a :class:TextCNNClassify`. It is used to instantiate a
    CNN model according to the specified arguments, defining the model architecture.


    Args:
        conv_dim (:obj:`int`, `optional`, defaults to 100):
            The output dimemsion of the convolution layer
        kernal_sizes (:obj:`string`, `optional`, defaults to 1,2,3,4):
            Specify the number of convolutional layers and kerval size for each layer.
        linear_hidden_size (:obj:`int`, `optional`, defaults to 512):
            number of neurals for fead-forward layers after each convolutional layer
        embed_size (:obj:`int`, `optional`, defaults to 300):
           embedding dimension for input tokens
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the CNN model.The defalut setting is to use BERTTokenizer so the vocab size is 30522 for
            english tasks.
        sequence_length (:obj:`int`, `optional`, defaults to 128):
           max sequence length for of the input text
    Examples::

        >>> from easynlp.modelzoo.models.cnn import TextCNNConfig
        >>> from easynlp.appzoo.classification import CNNTextClassify

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = TextCNNConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = CNNTextClassify(configuration)
    """
    model_type = "cnn"
    def __init__(self, conv_dim=100, kernel_sizes=[1,2,3], embed_size=300, vocab_size=21128, sequence_length=128, linear_hidden_size=None, **kwargs):
        super(TextCNNConfig, self).__init__()
        self.conv_dim = conv_dim
        self.kernel_sizes = kernel_sizes
        if linear_hidden_size:
            self.hidden_size = linear_hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> "PretrainedConfig":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        config = cls(conv_dim=config_dict['conv_dim'], kernel_sizes=config_dict['kernel_sizes'],
                     linear_hidden_size=config_dict['hidden_size'], embed_size=config_dict['embed_size'],
                     vocab_size=config_dict['vocab_size'], sequence_length=config_dict['sequence_length'])

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
