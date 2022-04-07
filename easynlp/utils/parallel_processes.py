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
import time
from collections import defaultdict

from .io_utils import io
from .logger import logger

try:
    from easy_predict import Process, DataFields
except:
    pass


def parse_row_by_schema(row, input_schema):
    row_dict = dict()
    for schema, content in zip(input_schema.split(','),
                               row.strip('\n').split('\t')):
        col_name, col_type, col_length = schema.split(':')
        if col_type == 'str':
            row_dict[col_name] = content
        elif col_type == 'int':
            if col_length == 1:
                row_dict[col_name] = int(content)
            else:
                row_dict[col_name] = [int(t) for t in content]
        elif col_type == 'float':
            if col_length == 1:
                row_dict[col_name] = float(content)
            else:
                row_dict[col_name] = [float(t) for t in content]
        else:
            raise RuntimeError('Invalid schema: %s' % schema)
    return row_dict


class SelfDefinedFileReaderProcess(Process):
    def __init__(self,
                 input_file,
                 input_schema,
                 job_name,
                 input_queue=None,
                 output_queue=None,
                 batch_size=1):
        super(SelfDefinedFileReaderProcess,
              self).__init__(job_name, 1, input_queue, output_queue,
                             batch_size)
        self.input_reader = io.open(input_file)
        self.input_schema = input_schema

    def process(self, in_data):
        for line in self.input_reader:
            input_dict = parse_row_by_schema(line, self.input_schema)
            self.put(input_dict)
        raise IndexError('Read tabel done')

    def close(self):
        self.input_reader.close()


class SelfDefinedFileWriterProcess(Process):
    def __init__(self,
                 output_file,
                 output_col_names,
                 job_name,
                 input_queue=None,
                 output_queue=None,
                 batch_size=1):
        super(SelfDefinedFileWriterProcess,
              self).__init__(job_name, 1, input_queue, output_queue,
                             batch_size)
        self.output_writer = io.open(output_file, 'w')
        self.output_cols = output_col_names
        self.cnt = 0
        self.start_time = time.time()

    def process(self, in_data):
        data = in_data[DataFields.result_to_save]
        out_list = [data[key] for key in self.output_cols]
        self.output_writer.write('\t'.join(out_list) + '\n')
        if self.cnt and self.cnt % 100 == 0:
            logger.info(
                '{} samples have been processed, time {:.3f}s  speed {:.3f}s/per sample'
                .format(self.cnt,
                        time.time() - self.start_time,
                        (time.time() - self.start_time) / self.cnt))
        self.cnt += 1

    def close(self):
        self.output_writer.close()


class SelfDefinedPredictorProcess(Process):
    def __init__(self,
                 predictor,
                 job_name,
                 thread_num,
                 input_queue=None,
                 output_queue=None,
                 batch_size=1,
                 mode='predict'):
        super(SelfDefinedPredictorProcess,
              self).__init__(job_name, thread_num, input_queue, output_queue,
                             batch_size)
        self.predictor = predictor
        self.mode = mode

    def process(self, in_data):
        if self.mode == 'preprocess':
            for record in in_data:
                for key, val in record.items():
                    if isinstance(val, bytes):
                        record[key] = val.decode('utf-8')
                    elif val is None:
                        record[key] = str(val)
            tmp = self.predictor.preprocess(in_data)
            rst = defaultdict(list)
            for key, val in tmp.items():
                rst[key] = val
            for record in in_data:
                for key, val in record.items():
                    if key not in tmp:
                        rst[key].append(val)
            self.put(rst)
        elif self.mode == 'predict':
            rst = self.predictor.predict(in_data)
            for key, val in rst.items():
                rst[key] = val
            for key, val in in_data.items():
                if key not in rst:
                    rst[key] = val
            self.put(rst)
        elif self.mode == 'postprocess':
            rst = self.predictor.postprocess(in_data)
            for b in range(len(rst)):
                for key, val in in_data.items():
                    if key not in rst[b]:
                        rst[b][key] = val[b]

            for item in rst:
                self.put(item)
        else:
            raise NotImplementedError


class SelfDefineTableFormatProcess(Process):
    def __init__(self,
                 input_queue,
                 output_queue,
                 reserved_col_names=[],
                 output_col_names=[]):
        """
        extract data from dict to preprare result dict which will be written to table or oss file
        Args:
          reserved_col_names list of str column names for reserved col
          output_col_names  list of column name for output table
          input_queue  the python queue for input
          output  the python queue for output
          float_digits: The number of digits of precision when writing floats out
        Return:
          push a k-v pair {'result_to_save': result_dict} into data_dict and return
          a result dict or a tuple, or a list of tuple or a list of dict
            if dict is returned, each key should be the same as column name of output table
            if tuple is returned, this tuple corespond to one table record
            if list of tuple is returned, they correspond to a list of table record
            if list of dict is returned, each element in list will be automatically converted to table record in table writer
        """
        super(SelfDefineTableFormatProcess,
              self).__init__('SelfDefineTableFormatProcess',
                             1,
                             input_queue=input_queue,
                             output_queue=output_queue)
        self.reserved_cols = reserved_col_names
        self.output_col_names = output_col_names

    def process(self, input_data):
        result_record_dict = dict()
        # write reserved cols
        for col in self.reserved_cols + self.output_col_names:
            if col not in input_data:
                continue
            if isinstance(input_data[col], dict) or isinstance(
                    input_data[col], list):
                result_record_dict[col] = json.dumps(input_data[col])
            else:
                result_record_dict[col] = input_data[col]
        # result should be returned with key DataFields.result_to_save
        input_data[DataFields.result_to_save] = result_record_dict
        return input_data
