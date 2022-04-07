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

import traceback

from ..appzoo.dataset import BaseDataset
from ..utils import parse_row_by_schema
from ..utils.logger import logger


class DistillatoryBaseDataset(BaseDataset):
    """A dataset class for supporting knowledge distillation. This class does not contain methods in :class:`BaseDataset` and only handles arguments that are proprietary to knowledge distillation.

    Args:
        user_defined_parameters:
            The dict of user defined parameters for knowledge distillation.
    """
    def __init__(self, user_defined_parameters: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if user_defined_parameters is None:
            raise ValueError

        if not isinstance(user_defined_parameters, dict):
            raise TypeError(
                '`user_defined_parameters` should be a characterized '
                'dictionary data structure.')
        try:
            distill_params = user_defined_parameters.get('app_parameters')
            self.kd_type = distill_params['type']
            self.teacher_rows = self.readlines_from_file(
                distill_params['logits_saved_path'], skip_first_line=False)
            self.logits_name = distill_params['logits_name']
        except KeyError:
            traceback.print_exc()
            logger.error(
                'For knowledge distillation, the parameters should be passed '
                'in via user_defined_parameters.app_parameters and should contain '
                'type, logits_saved_path and logits_name.')
            exit(-1)

    def __getitem__(self, item):
        """Obtaining the next data item."""

        row = self.prepare_row(item)
        if hasattr(self, 'teacher_rows'):
            teacher_row = self.teacher_rows[item].strip('\n')
            row += ('\t' + teacher_row)

        if self.output_format == 'dict' and self.input_schema:
            row = parse_row_by_schema(row, self.input_schema)
        try:
            return self.convert_single_row_to_example(row)
        except:
            logger.info('Failed row: {}'.format(row))
            raise RuntimeError

    def convert_single_row_to_example(self, row):
        """Converting the examples into the dict of values."""

        encode = super().convert_single_row_to_example(row)
        encode['teacher_logits'] = row[self.logits_name]
        return encode

    def load_bin_file(filepath: str):
        pass
