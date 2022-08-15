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
import torch
import numpy as np
import albumentations
from io import BytesIO
import base64
from PIL import Image, ImageFile

from .tokenizer import ImageTextBERTTokenizer, ImageTextGPT2Tokenizer
from .clip import _transform as build_clip_image_transform
from ..dataset import BaseDataset
from ...utils import get_pretrain_model_path
from ...modelzoo import AutoConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_general_image_transform(target_image_size, random_crop):
    """
        general image transform
    """
    rescaler = albumentations.Resize(height = target_image_size, width = target_image_size)
    if not random_crop:
        cropper = albumentations.CenterCrop(height = target_image_size, width = target_image_size)
    else:
        cropper = albumentations.RandomCrop(height = target_image_size, width = target_image_size)
    
    preprocessor = albumentations.Compose([rescaler, cropper])

    return preprocessor
    

class CLIPGPTImageTextDataset(BaseDataset):
    """
    Classification Dataset

    Args:
        pretrained_model_name_or_path: to init tokenizer.
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
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 second_sequence=None,
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)

        # when pretraining, the pretrained_model_name_or_path is actually the tokenizer
        # if we don't set pretrained_model_name_or_path, \
        # the vocab.txt will not be automatically copied to the new director of pretrained model
        # Therefore, the pretrained_model_name_or_path should be set in pretraining
        self.enable_pretraining = False
        if pretrained_model_name_or_path is None:
            self.enable_pretraining = True
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if config.model_type in ['bert', 'gpt2']:
                self.enable_pretraining = True

        # image size and sequence length
        self.img_size = int(user_defined_parameters.get('img_size', 224))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.max_seq_length = max_seq_length
        assert self.max_seq_length == (self.img_len + self.text_len), "max_seq_length thould be equal to the sum of img_seq and text_seq"
        
        # text tokenizer
        if self.enable_pretraining:
            text_tokenizer_path = pretrained_model_name_or_path if pretrained_model_name_or_path else 'bert-base-chinese'
            text_tokenizer_path = get_pretrain_model_path(text_tokenizer_path)
        else:
            text_tokenizer_path = pretrained_model_name_or_path
        self.tokenizer = ImageTextBERTTokenizer(text_tokenizer_path, start_id = 0)

        # image encoder type
        if self.enable_pretraining:
            # if image_encoder is not parametered, default image encoder is 'vit'
            is_clip_encoder = user_defined_parameters.get('app_parameters', True).get('enable_vit', True)
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if config.prefix_encoder_type == 'vit':
                is_clip_encoder = True
            else:
                is_clip_encoder = False
        
        assert is_clip_encoder == True, 'wrong image encode settings'


        # image preprocessor
        self.preprocessor = build_clip_image_transform(self.img_size)
        
        # first_sequence and second_sequence
        self.first_sequence = first_sequence
        if second_sequence:
            assert second_sequence in self.column_names, \
                "Column name %s needs to be included in columns" % second_sequence
            self.second_sequence = second_sequence
        else:
            self.second_sequence = None
        
        # image input format
        self.enable_img_path = True if user_defined_parameters.get('enable_img_path', 'false').lower() == 'true' else False
        self.img_root_dir = user_defined_parameters.get('img_root_dir', None)
        if self.enable_img_path:
            assert self.img_root_dir is not None, '\'img_root_dir\' should have value'


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
        encoding = {}
        image_str = row[self.first_sequence]
        text = row[self.second_sequence] if self.second_sequence else None

        # preprocess image
        if self.enable_img_path:
            image = Image.open(os.path.join(self.img_root_dir, image_str))
        else:
            image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str)))
        image = self.preprocessor(image)    # type=torch.Tensor , shape = (3, img_size, img_size)
        # print ("vit_image=", image.shape)

        encoding['image'] = image.tolist()

        # preprocess text
        text_ids = self.tokenizer.encode(text)
        text_ids = text_ids[: self.text_len]
        n_pad = self.text_len - len(text_ids)
        text_ids += [self.tokenizer.end_token_id] * n_pad
        encoding['text'] = text_ids        # type=list

        return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}

class VQGANGPTImageTextDataset(BaseDataset):
    """
    Classification Dataset

    Args:
        pretrained_model_name_or_path: to init tokenizer.
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
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 second_sequence=None,
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)

        # when pretraining, the pretrained_model_name_or_path is actually the tokenizer
        # if we don't set pretrained_model_name_or_path, \
        # the vocab.txt will not be automatically copied to the new director of pretrained model
        # Therefore, the pretrained_model_name_or_path should be set in pretraining
        self.enable_pretraining = False
        if pretrained_model_name_or_path is None:
            self.enable_pretraining = True
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if config.model_type in ['bert', 'gpt2']:
                self.enable_pretraining = True

        # image size and sequence length
        self.img_size = int(user_defined_parameters.get('img_size', 256))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.max_seq_length = max_seq_length
        assert self.max_seq_length == (self.img_len + self.text_len), "max_seq_length thould be equal to the sum of img_seq and text_seq"
        
        # text tokenizer
        if self.enable_pretraining:
            text_tokenizer_path = pretrained_model_name_or_path if pretrained_model_name_or_path else 'bert-base-chinese'
            text_tokenizer_path = get_pretrain_model_path(text_tokenizer_path)
        else:
            text_tokenizer_path = pretrained_model_name_or_path
        self.tokenizer = ImageTextBERTTokenizer(text_tokenizer_path, start_id = 0)

        # image encoder type
        if self.enable_pretraining:
            # if image_encoder is not parametered, default image encoder is 'vit', not 'vqgan'
            is_vqgan_encoder = user_defined_parameters.get('app_parameters', False).get('enable_vqgan', False)
        else:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            if config.prefix_encoder_type == 'vqgan':
                is_vqgan_encoder = True
            elif config.model_type == 'artist_i2t':
                # for old version artist_i2t pretrained model
                is_vqgan_encoder = True
            else:
                is_vqgan_encoder = False
            
        
        assert is_vqgan_encoder == True, 'wrong image encode settings'


        # image preprocessor
        random_crop = bool(user_defined_parameters.get('random_crop', False))
        self.preprocessor = build_general_image_transform(self.img_size, random_crop)
        
        # first_sequence and second_sequence
        self.first_sequence = first_sequence
        if second_sequence:
            assert second_sequence in self.column_names, \
                "Column name %s needs to be included in columns" % second_sequence
            self.second_sequence = second_sequence
        else:
            self.second_sequence = None
        

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
        encoding = {}
        image_str = row[self.first_sequence]
        text = row[self.second_sequence] if self.second_sequence else None

        # preprocess image
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str))).convert("RGB")
        image = np.array(image).astype(np.uint8) 
        image = self.preprocessor(image=image)["image"]   # type=numpy.array
        image = (image/127.5 - 1.0).astype(np.float32)    # shape = (img_size, img_size, 3)
        image = image.transpose((2,0,1))                  # shape = (3, img_size, img_size)
        # print ("vqgan_image=", image.shape)

        encoding['image'] = image.tolist()

        # preprocess text
        text_ids = self.tokenizer.encode(text)
        text_ids = text_ids[: self.text_len]
        n_pad = self.text_len - len(text_ids)
        text_ids += [self.tokenizer.end_token_id] * n_pad
        encoding['text'] = text_ids        # type=list

        return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}