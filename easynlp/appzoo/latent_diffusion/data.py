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


import numpy as np
import torch
import albumentations
from io import BytesIO
import base64
from PIL import Image, ImageFile
import json

from ..dataset import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LdmDataset(BaseDataset):
    """
    latent diffusion Generation Dataset

    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file.
        max_seq_length: max sequence length of each input instance.
        input schema: columns of input data file
        first_sequence: input text
        second_sequence: base64 data of the image
    """
    def __init__(self,
                 data_file,
                 max_seq_length,
                 input_schema,
                 first_sequence,
                pretrained_model_name_or_path,
                 second_sequence=None,
                 user_defined_parameters=None,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)
        with open(pretrained_model_name_or_path+'/config.json','r') as config_handle:
            self.raw_config=json.load(config_handle)
        self.size = self.raw_config["model"]["params"]['first_stage_config']['params']['ddconfig']['resolution']
        self.random_crop = bool(user_defined_parameters.get('random_crop', False))
        self.max_seq_length = max_seq_length


        # image preprocessor
        self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

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
        text = row[self.first_sequence]
        image_str = row[self.second_sequence] if self.second_sequence else None

        # caption
        encoding['caption'] = text

        # preprocess image
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str))).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        encoding['image'] = image
        return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        batch_data = {}
        image = torch.as_tensor(np.array([example['image'] for example in features]))
        caption = [example['caption'] for example in features]
        # print(words_mat.shape, words_emb.shape)
        batch_data['image'] = image
        batch_data['caption'] = caption
        return batch_data
