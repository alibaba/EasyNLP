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

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from utils import io
from utils.logger import logger


class BaseDataset(Dataset):
    def __init__(self, data_file,
                 skip_first_line=False,
                 selected_columns="",
                 reader_buffer_size=256,
                 input_schema=None,
                 is_training=True,
                 output_format="line",
                 *args,
                 **kwargs):
        self.data_source = self.identify_data_source(data_file)
        self.input_schema = input_schema
        self.output_format = output_format
        assert self.output_format in ["line", "dict"]
        if self.data_source in ["local", "oss"]:
            self.skip_first_line = skip_first_line
            self.data_rows = self.readlines_from_file(data_file)
        elif self.data_source == "odps":
            import common_io
            slice_id = 0
            slice_count = 1
            if dist.is_initialized():
                slice_id = dist.get_rank()
                slice_count = dist.get_world_size()
            if not is_training:
                slice_id = 0
                slice_count = 1
            self.table_reader = common_io.table.TableReader(
                data_file,
                selected_cols=selected_columns,
                slice_id=slice_id,
                slice_count=slice_count, capacity=reader_buffer_size)
            self.slice_id = slice_id
            self.table_row_count = self.table_reader.get_row_count()
            self.start_position = self.table_row_count * slice_id
            self.input_schema = self.get_odps_input_schema()
        else:
            raise NotImplementedError
        if self.input_schema:
            self.column_names = [t.split(":")[0] for t in self.input_schema.split(",")]
        else:
            self.column_names = list()
        self.cnt = 0

    def __del__(self):
        if self.data_source == "odps":
            self.table_reader.close()

    def __getitem__(self, item):
        if self.data_source in ["local", "oss"]:
            row = self.data_rows[item]

        elif self.data_source == "odps":
            # TODO: Add a buffer to support random shuffle
            try:
                row = self.table_reader.read(1)
                self.cnt += 1
            except Exception:
                print("[Pid %d] Enter the end of the table, %d sample processed, seek start position %d" %
                      (self.slice_id, self.cnt, self.start_position))
                self.table_reader.seek(self.start_position)
                row = self.table_reader.read(1)
                self.cnt = 1
                print("[Pid %d] Read success" % self.slice_id)
            row = "\t".join([t.decode("utf-8") for t in row[0]])
        else:
            raise NotImplementedError
        if self.output_format == "dict" and self.input_schema:
            row = self.parse_row_by_schema(row, self.input_schema)
        try:
            return self.convert_single_row_to_example(row)
        except:
            logger.info("Failed row: {}".format(row))
            raise RuntimeError

    def __len__(self):
        if self.data_source in ["local", "oss"]:
            return len(self.data_rows)
        elif self.data_source == "odps":
            return self.table_row_count
        else:
            raise NotImplementedError

    def identify_data_source(self, data_file):
        if "odps://" in data_file:
            data_source = "odps"
        elif "oss://" in data_file:
            data_source = "oss"
        else:
            data_source = "local"
        return data_source

    def readlines_from_file(self, data_file):
        i = 0
        with io.open(data_file) as f:
            if self.skip_first_line:
                f.readline()
            if data_file.index('json') != -1:
                task_rows = eval(json.load(f))['data']
            # data_rows = f.readlines()
            # if i % 100000:
            #     logger.info("{} lines read from {}".format(i, data_file))
        # task, content, label = items[0], items[1], items[2]
        with open('/apsarapangu/disk3/zhangtaolin.ztl/MeLL_pytorch/data/all_data.json', 'r') as file:
            all_data = eval(json.load(file))['data']
            all_data_dict = {}
            for item in all_data:
                taskKey = item['taskKey']
                if taskKey not in all_data_dict.keys():
                    all_data_dict[taskKey] = item['dataset']
        data_rows = []
        if type(task_rows) == dict:
            task_rows = [task_rows]
        for item in task_rows:
            taskKey = item['taskKey']
            dataset = all_data_dict[taskKey]
            for dataset_i in dataset:
                text = dataset_i['text']
                label = dataset_i['label']
                data_rows.append(
                    taskKey + '\t' + text + '\t' + label
                )
        return data_rows

    def readlines_from_odps(self, data_file):
        raise NotImplementedError

    @staticmethod
    def parse_row_by_schema(row, input_schema):
        row_dict = dict()
        for schema, content in zip(input_schema.split(","), row.strip().split("\t")):
            col_name, col_type, col_length = schema.split(":")
            if col_type == "str":
                row_dict[col_name] = content
            elif col_type == "int":
                if col_length == 1:
                    row_dict[col_name] = int(content)
                else:
                    row_dict[col_name] = [int(t) for t in content]
            elif col_type == "float":
                if col_length == 1:
                    row_dict[col_name] = float(content)
                else:
                    row_dict[col_name] = [float(t) for t in content]
            else:
                raise RuntimeError("Invalid schema: %s" % schema)
        return row_dict

    @property
    def eval_metrics(self):
        return ('accuracy',)

    @property
    def label_enumerate_values(self):
        return ["0", "1"]

    def batch_fn(self, features):
        raise NotImplementedError

    def convert_single_row_to_example(self, row):
        raise NotImplementedError

    def get_odps_input_schema(self):
        schemas = self.table_reader.get_schema()
        colname2schema = dict()
        for col_name, odps_type, _ in schemas:
            if odps_type == u"string":
                colname2schema[str(col_name)] = "str"
            elif odps_type == u"double":
                colname2schema[str(col_name)] = "float"
            elif odps_type == u"bigint":
                colname2schema[str(col_name)] = "int"
            else:
                colname2schema[str(col_name)] = "str"

        col_with_schemas = ["{}:{}:1".format(col_name, colname2schema[col_name])
                            for col_name, _, _ in schemas]

        rst_schema = ",".join(col_with_schemas)
        print("Input Schema: ", rst_schema)
        return rst_schema


class LogitsDataset(BaseDataset):
    def __init__(self, data_file, label_enumerate_values, eval_metrics, **kwargs):
        super(LogitsDataset, self).__init__(data_file, **kwargs)
        self._label_enumerate_values = label_enumerate_values
        self._eval_metrics = eval_metrics

    @property
    def eval_metrics(self):
        return self._eval_metrics

    @property
    def label_enumerate_values(self):
        return self._label_enumerate_values

    def batch_fn(self, features):
        inputs = {
            "teacher_logits": torch.tensor([f[0] for f in features], dtype=torch.float),
            "input_ids": torch.tensor([f[1] for f in features], dtype=torch.long),
            "input_mask": torch.tensor([f[2] for f in features], dtype=torch.long),
            "segment_ids": torch.tensor([f[3] for f in features], dtype=torch.long),
            "label_ids": torch.tensor([f[4] for f in features], dtype=torch.long),
        }
        return inputs

    def convert_single_row_to_example(self, row):
        logits, input_ids, input_mask, segment_ids, label_id = row.strip().split("\t")
        logits = [float(t) for t in logits.split(",")]
        input_ids = [int(t) for t in input_ids.split(",")]
        input_mask = [int(t) for t in input_mask.split(",")]
        segment_ids = [int(t) for t in segment_ids.split(",")]
        try:
            label_id = int(label_id)
        except:
            label_id = float(label_id)

        return (logits, input_ids, input_mask, segment_ids, label_id)