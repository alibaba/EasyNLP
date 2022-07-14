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
from .tokenizer import ArtistBERTTokenizer, ArtistGPT2Tokenizer
#from ...utils import io
from ..dataset import BaseDataset
import numpy as np
import albumentations
from io import BytesIO
import base64
from PIL import Image, ImageFile
from easynlp.utils import get_pretrain_model_path
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextDataset(BaseDataset):
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
        # image size and sequence length
        self.img_size = int(user_defined_parameters.get('img_size', 256))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.max_seq_length = max_seq_length
        assert self.max_seq_length == (self.img_len + self.text_len), "max_seq_length thould be equal to the sum of img_seq and text_seq"
        
        # text tokenizer
        if pretrained_model_name_or_path == None:
            text_tokenizer_path = get_pretrain_model_path(user_defined_parameters.get('text_tokenizer', 'bert-base-chinese'))
        else:
            text_tokenizer_path = pretrained_model_name_or_path
        self.tokenizer = ArtistBERTTokenizer(text_tokenizer_path, start_id = 0)
        
        # image preprocessor
        self.random_crop = bool(user_defined_parameters.get('random_crop', False))
        self.preprocessor = self.setup_transform(self.img_size, self.random_crop)


        self.first_sequence = first_sequence
        if second_sequence:
            assert second_sequence in self.column_names, \
                "Column name %s needs to be included in columns" % second_sequence
            self.second_sequence = second_sequence
        else:
            self.second_sequence = None
        
    def setup_transform(self, target_image_size, random_crop):
        """
            image transform
        """
        rescaler = albumentations.SmallestMaxSize(max_size = target_image_size)
        if not random_crop:
            cropper = albumentations.CenterCrop(height = target_image_size, width = target_image_size)
        else:
            cropper = albumentations.RandomCrop(height = target_image_size, width = target_image_size)

        preprocessor = albumentations.Compose([rescaler, cropper])

        return preprocessor

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
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        encoding['image'] = image

        # preprocess text
        text_ids = self.tokenizer.encode(text)
        text_ids = text_ids[: self.text_len]
        n_pad = self.text_len - len(text_ids)
        text_ids += [self.tokenizer.end_token_id] * n_pad
        encoding['text'] = np.array(text_ids)

        return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}