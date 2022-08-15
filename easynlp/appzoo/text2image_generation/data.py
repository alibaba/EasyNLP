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
import albumentations
from io import BytesIO
import base64
from PIL import Image, ImageFile

from ...modelzoo import AutoTokenizer
from ...utils import io
from ..dataset import BaseDataset
from ...utils import get_pretrain_model_path

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TextImageDataset(BaseDataset):
    """
    Text to Image Generation Dataset

    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file.
        max_seq_length: max sequence length of each input instance.
        input schema: columns of input data file
        first_sequence: input text
        second_sequence: base64 data of the image
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
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = get_pretrain_model_path('hfl/chinese-roberta-wwm-ext')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.size = int(user_defined_parameters.get('size', 256))
        self.random_crop = bool(user_defined_parameters.get('random_crop', False))
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.text_vocab_size = len(self.tokenizer)
        self.img_vocab_size = int(user_defined_parameters.get('img_vocab_size', 16384))
        self.max_seq_length = max_seq_length
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')

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

        # preprocess text
        text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        text_ids = text_ids[: self.text_len]
        n_pad = self.text_len - len(text_ids)
        text_ids += [self.pad_id] * n_pad
        encoding['text'] = np.array(text_ids) + self.img_vocab_size

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
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}


class TextImageKnowlDataset(BaseDataset):
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
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = get_pretrain_model_path('hfl/chinese-roberta-wwm-ext')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        self.size = int(user_defined_parameters.get('size', 256))
        self.random_crop = bool(user_defined_parameters.get('random_crop', False))
        self.text_len = int(user_defined_parameters.get('text_len', 32))
        self.img_len = int(user_defined_parameters.get('img_len', 256))
        self.text_vocab_size = len(self.tokenizer)
        self.img_vocab_size = int(user_defined_parameters.get('img_vocab_size', 16384))
        self.max_seq_length = max_seq_length
        self.pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        self.entity_num = int(user_defined_parameters.get('entity_num', 10))
        self.entity_emb_path = user_defined_parameters.get('entity_emb_path')

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
        
        pt = torch.load(self.entity_emb_path)
        self.embed = torch.nn.Embedding.from_pretrained(pt)
        
    
    def convert_single_row_to_example(self, row):
        encoding = {}
        text = row[self.first_sequence]
        image_str = row[self.second_sequence] if self.second_sequence else None
        lex_id = row['lex_id']
        pos_s = row['pos_s']
        pos_e = row['pos_e']
        token_len = int(row['token_len'])

        # preprocess text
        text_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        text_ids = text_ids[: self.text_len]
        n_pad = self.text_len - len(text_ids)
        text_ids += [self.pad_id] * n_pad
        encoding['text'] = np.array(text_ids) + self.img_vocab_size

        # preprocess image
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_str))).convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        encoding['image'] = image

        # preprocess word_matrix
        words_mat = np.zeros([self.entity_num, self.text_len], dtype=np.int)
        if len(lex_id) > 0:
            ents = lex_id.split(' ')[:self.entity_num]
            pos_s = [int(x) for x in pos_s.split(' ')]
            pos_e = [int(x) for x in pos_e.split(' ')]
            ent_pos_s = pos_s[token_len:token_len+self.entity_num]
            ent_pos_e = pos_e[token_len:token_len+self.entity_num]

            for i, ent in enumerate(ents):
                words_mat[i, ent_pos_s[i]:ent_pos_e[i]+1] = ent
        encoding['words_mat'] = words_mat
        return encoding


    def batch_fn(self, batch):
        """
            Divide examples into batches.
        """
        batch_data = {}
        image = torch.as_tensor([example['image'] for example in batch])
        text = torch.LongTensor([example['text'] for example in batch])
        words_mat = torch.LongTensor([example['words_mat'] for example in batch])
        words_emb = self.embed(words_mat)
        # print(words_mat.shape, words_emb.shape)
        batch_data['image'] = image
        batch_data['text'] = text
        batch_data['words_emb'] = words_emb
        return batch_data