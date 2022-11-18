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

import torch
import numpy as np
from ...modelzoo import BertTokenizer
from .bert_tokenizer import FullTokenizer
from ...utils import io
from ..dataset import BaseDataset
# from ...modelzoo.models.clip.processing_clip import CLIPProcessor
from PIL import Image
import base64
from io import BytesIO
from typing import Union, List, Tuple
from ...utils import losses, get_pretrain_model_path, get_args

def _center_crop(image, size):
    """
    Crops `image` to the given size using a center crop. Note that if the image is too small to be cropped to the
    size is given, it will be padded (so the returned result has the size asked).
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            The size to which crop the image.
    """

    if not isinstance(size, tuple):
        size = (size, size)

    image_width, image_height = image.size
    crop_height, crop_width = size

    crop_top = int((image_height - crop_height + 1) * 0.5)
    crop_left = int((image_width - crop_width + 1) * 0.5)

    return image.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

def _resize(image, size, resample=Image.BICUBIC):
    """
    Resizes `image`. Note that this will trigger a conversion of `image` to a PIL Image.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to resize.
        size (`int` or `Tuple[int, int]`):
            The size to use for resizing the image. If `int` it will be resized to match the shorter side
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            The filter to user for resampling.
    """

    if isinstance(size, tuple):
        new_w, new_h = size
    else:
        width, height = image.size
        short, long = (width, height) if width <= height else (height, width)
        if short == size:
            return image
        new_short, new_long = size, int(size * long / short)
        new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
    return image.resize((new_w, new_h), resample)  

def _to_numpy_array(image, rescale=None, channel_first=True):
    """
    Converts `image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
    dimension.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to convert to a NumPy array.
        rescale (`bool`, *optional*):
            Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
            default to `True` if the image is a PIL Image or an array/tensor of integers, `False` otherwise.
        channel_first (`bool`, *optional*, defaults to `True`):
            Whether or not to permute the dimensions of the image to put the channel dimension first.
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    if rescale is None:
        rescale = isinstance(image.flat[0], np.integer)

    if rescale:
        image = image.astype(np.float32) / 255.0

    if channel_first and image.ndim == 3:
        image = image.transpose(2, 0, 1)

    return image

def _normalize(image, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]):
    """
    Normalizes `image` with `mean` and `std`. Note that this will trigger a conversion of `image` to a NumPy array
    if it's a PIL Image.
    Args:
        image (`PIL.Image.Image` or `np.ndarray` or `torch.Tensor`):
            The image to normalize.
        mean (`List[float]` or `np.ndarray` or `torch.Tensor`):
            The mean (per channel) to use for normalization.
        std (`List[float]` or `np.ndarray` or `torch.Tensor`):
            The standard deviation (per channel) to use for normalization.
    """

    if isinstance(image, Image.Image):
        image = _to_numpy_array(image)

    if isinstance(image, np.ndarray):
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean).astype(image.dtype)
        if not isinstance(std, np.ndarray):
            std = np.array(std).astype(image.dtype)
    elif is_torch_tensor(image):
        import torch

        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)

    if image.ndim == 3 and image.shape[0] in [1, 3]:
        return (image - mean[:, None, None]) / std[:, None, None]
    else:
        return (image - mean) / std


class WukongCLIPDataset(BaseDataset):
    """
    Classification Dataset
    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file.
        max_seq_length: max sequence length of each input instance.
        first_sequence: input text
        label_name: label column name
        second_sequence: set as None
        label_enumerate_values: a list of label values
        multi_label: set as True if perform multi-label classification, otherwise False
    """

    def __init__(self,
                    pretrained_model_name_or_path,
                    data_file,
                    max_seq_length=32,
                    input_schema=None,
                    first_sequence=None,
                    label_name=None,
                    second_sequence=None,
                    label_enumerate_values=None,
                    user_defined_parameters=None,
                    *args,
                    **kwargs):

        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)

        self.text_col = first_sequence
        self.image_col=second_sequence
        pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
        self.tokenizer=FullTokenizer(vocab_file=pretrained_model_name_or_path+'/vocab.txt')
        self.max_text_length=max_seq_length
        self.do_resize=True
        self.size=224
        self.resample=Image.BICUBIC
        self.do_center_crop=True
        self.crop_size=224
        self.do_normalize=True

    def tokenize(self,texts: Union[str, List[str]], context_length: int = 32) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all baseline models use 32 as the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = []
        for text in texts:
            all_tokens.append([self.tokenizer.vocab['[CLS]']] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))[:context_length - 2] + [self.tokenizer.vocab['[SEP]']])
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            assert len(tokens) <= context_length
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result

    def convert_single_row_to_example(self, row):
        """Convert sample token to indices.
            Args:
                row: contains sequence and label.
                text_a: the first sequence in row.
                text_b: the second sequence in row if self.second_sequence is true.
                label: label token if self.label_name is true.
            Returns: sing example
                encoding: an example contains token indices.
        """
        _text=row[self.text_col]
        tk_result={}
        tk_result['input_ids'] = self.tokenize(_text)
        one_image= Image.open(BytesIO(base64.urlsafe_b64decode(row[self.image_col])))
        images=[one_image]
        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [_resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [_center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [_normalize(image=image) for image in images]
        return {'text':tk_result,'pixel_values':torch.tensor(images)}
    
    def batch_fn(self, features):
        # """
        #     Divide examples into batches.
        # """
        output={'pixel_values':[],'input_ids':[]}
        for dic in features:
            output['pixel_values'].append(dic['pixel_values'])
            output['input_ids'].append(dic['text']['input_ids'])

        output['pixel_values']=torch.cat(output['pixel_values'],dim=0)
        output['input_ids']=torch.cat(output['input_ids'],dim=0)
        return output
