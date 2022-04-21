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

from typing import List
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, dataset

from datasets import load_dataset as hf_load_dataset
from ast import literal_eval

from ..utils import io, parse_row_by_schema, get_dir_name
from ..utils.logger import logger
from ..modelzoo import AutoTokenizer


class BaseDataset(Dataset):

    def __init__(self,
                 data_file: str,
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
            self.get_odps_reader2(data_file, is_training, reader_buffer_size, kwargs)
            # self.get_odps_reader(data_file, is_training, reader_buffer_size, kwargs)
        else:
            raise NotImplementedError

        if self.input_schema:
            self.column_names = [t.split(":")[0] for t in self.input_schema.split(",")]
        else:
            self.column_names = list()
        self.cnt = 0

    def get_odps_reader(self, data_file, is_training, reader_buffer_size, kwargs=None):
        import common_io
        slice_id = 0
        slice_count = 1
        if dist.is_initialized():
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        if not is_training:
            slice_id = 0
            slice_count = 1
        self.table_reader = common_io.table.TableReader(data_file,
                                                        selected_cols=selected_columns,
                                                        slice_id=slice_id,
                                                        slice_count=slice_count,
                                                        capacity=reader_buffer_size)

        self.table_row_count = self.table_reader.get_row_count()
        self.start_position = self.table_row_count * slice_id
        prefetch_all = kwargs.get('prefetch_all', False)
        if prefetch_all:
            self.data_rows = []
            for i in range(self.table_row_count):
                row = self.table_reader.read(1)
                row = "\t".join([t.decode("utf-8") for t in row[0]])
                self.data_rows.append(row)

            self.table_reader.seek(self.start_position)

        self.slice_id = slice_id
        self.input_schema = self.get_odps_input_schema()

    def get_odps_reader2(self, table_path, is_training, reader_buffer_size, kwargs=None):
        import common_io
        slice_id = 0
        slice_count = 1
        if dist.is_initialized():
            slice_id = dist.get_rank()
            slice_count = dist.get_world_size()
        if not is_training:
            slice_id = 0
            slice_count = 1

        self.table_path = table_path
        self.table_reader = common_io.table.TableReader(table_path,
                                             slice_id=slice_id,
                                             slice_count=slice_count,
                                             num_threads=0)
        self.table_row_count = self.table_reader.get_row_count()
        self.start_position = self.table_row_count * slice_id
        self.end_pos = self.table_reader.end_pos        

        prefetch_all = kwargs.get('prefetch_all', False)
        if prefetch_all:
            self.data_rows = []
            for i in range(self.table_row_count):
                row = self.table_reader.read(1)
                row = "\t".join([t.decode("utf-8") for t in row[0]])
                self.data_rows.append(row)
        
        self.slice_id = slice_id
        self.input_schema = self.get_odps_input_schema()
        # super(TableDataset, self).__init__()
        print("table total_row_count:{}, start_pos:{}, end_pos:{}".format(
            self.table_row_count, self.start_position, self.end_pos))

        self.table_reader.close()
        self.table_reader = None

    def _get_slice_range(self, row_count, worker_info, baseline=0):
        if worker_info is None:
            worker_id = 0
            num_data_workers = 1
        else:
            worker_id = worker_info.id
            num_data_workers = worker_info.num_workers

        # div-mod split, each slice data count max diff 1
        size = int(row_count / num_data_workers)
        split_point = row_count % num_data_workers
        if worker_id < split_point:
            start = worker_id * (size + 1) + baseline
            end = start + (size + 1)
        else:
            start = split_point * (size + 1) + (worker_id - split_point) * size + baseline
            end = start + size
        return start, end
    
    def __del__(self):
        if self.data_source == "odps":
            if self.table_reader is not None:
                self.table_reader.close()

    def __getitem__(self, item):
        # row = self.prepare_row(item)

        if self.data_source in ["local", "oss"]:
            row = self.data_rows[item].strip('\n')
        elif self.data_source == "odps":
            if self.table_reader is None:            
                worker_info = torch.utils.data.get_worker_info()
                table_start, table_end = self._get_slice_range(self.table_row_count, worker_info, self.start_position)
                table_path = "{}?start={}&end={}".format(self.table_path, table_start, table_end)
                print("table_path:%s" % table_path)

                import common_io
                self.table_reader = common_io.table.TableReader(table_path, num_threads=1, capacity=1024)

            try:
                row = self.table_reader.read(num_records=1, allow_smaller_final_batch=True)
                self.cnt += 1
            except Exception:
                worker_info = torch.utils.data.get_worker_info()
                table_start, table_end = self._get_slice_range(self.table_row_count, worker_info, self.start_position)

                print(
                    "[Pid %d] Enter the end of the table, %d sample processed, seek start position %d"
                    % (self.slice_id, self.cnt, table_start))
                self.table_reader.seek(table_start)
                row = self.table_reader.read(num_records=1, allow_smaller_final_batch=True)
                self.cnt = 1
                print("[Pid %d] Read success" % self.slice_id)

            row = "\t".join([t.decode("utf-8") for t in row[0]])
        else:
            raise NotImplementedError

        if self.output_format == 'dict' and self.input_schema:
            row = parse_row_by_schema(row, self.input_schema)
        try:
            return self.convert_single_row_to_example(row)
        except :
            logger.info("Failed row: {}".format(row))
            raise RuntimeError

    def __len__(self):
        if self.data_source in ["local", "oss"]:
            return len(self.data_rows)
        elif self.data_source == "odps":
            return self.table_row_count
        else:
            raise NotImplementedError

    def prepare_row(self, item: int) -> str:
        if self.data_source in ["local", "oss"]:
            row = self.data_rows[item].strip('\n')
        elif self.data_source == "odps":
            # TODO: Add a buffer to support random shuffle
            try:
                row = self.table_reader.read(1)
                self.cnt += 1
            except Exception:
                print(
                    "[Pid %d] Enter the end of the table, %d sample processed, seek start position %d"
                    % (self.slice_id, self.cnt, self.start_position))
                self.table_reader.seek(self.start_position)
                row = self.table_reader.read(1)
                self.cnt = 1
                print("[Pid %d] Read success" % self.slice_id)
            row = "\t".join([t.decode("utf-8") for t in row[0]])
        else:
            raise NotImplementedError

        return row

    def identify_data_source(self, data_file):
        if "odps://" in data_file:
            data_source = "odps"
        elif "oss://" in data_file:
            data_source = "oss"
        else:
            data_source = "local"
        return data_source

    def readlines_from_file(self, data_file, skip_first_line=None) -> List[str]:
        i = 0
        if skip_first_line is None:
            skip_first_line = self.skip_first_line
        with io.open(data_file) as f:
            if skip_first_line:
                f.readline()
            data_rows = f.readlines()
            if i % 100000:
                logger.info("{} lines read from {}".format(i, data_file))
        return data_rows

    def readlines_from_odps(self, data_file):
        raise NotImplementedError

    @property
    def label_enumerate_values(self):
        return ["0", "1"]

    def batch_fn(self, features):
        raise NotImplementedError

    @property
    def labels(self):
        labels = []
        for row in self.data_rows:
            row = parse_row_by_schema(row, self.input_schema)
            label = row[self.label_name] if self.label_name else None
            labels.append(int(label))
        return np.asarray(labels)

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

        col_with_schemas = [
            "{}:{}:1".format(col_name, colname2schema[col_name]) for col_name, _, _ in schemas
        ]

        rst_schema = ",".join(col_with_schemas)
        print("Input Schema: ", rst_schema)
        return rst_schema

class DataProcessor(Dataset):
    def __init__(self, 
                 datasets:dataset, 
                 pretrained_model_name_or_path:str,
                 max_seq_length:int,
                 sentence_col_names:str=None,
                 label_col_name:str=None,
                 label_enumerate_value:str=None,
                 ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.max_seq_length = max_seq_length
        self.sent1 = self.sent2 = self.label_col_name = self.num_label = self.label_enumerate_value \
            = self.label_mapping = None
        # Loaded data is from huggingface or EASYNLP
        dataset_info = getattr(datasets, "info", None)
        assert dataset_info is not None, "DataProcessor supports only huggingface_datasets input."
        self.data_sourse = "hub" if dataset_info.description != "" else "user"
        self.datasets = datasets
        if self.data_sourse == "hub":
            data_features = dataset_info.features
            feature_list = list(data_features)
            self.sent1 = feature_list[0]
            if feature_list[1] != "label":
                self.sent2 = feature_list[1]
            else:
                self.sent2 = None
            self.label_col_name = "label"
            self.num_label = data_features["label"].num_classes
            self.label_enumerate_value = data_features["label"].names
            self.label_mapping = {label: i for i, label in enumerate(self.label_enumerate_value)}
        # Loaded data is user-defined
        else:
            self.label_col_name = label_col_name.replace(" ", "")
            assert self.label_col_name in datasets.features, f"Can't find the col name of {self.label_col_name}. \
                When using a custom dataset, you need to specify all optional parameters"
            sentence_col_name_list = sentence_col_names.replace(" ", "").split(",")
            self.sent1 = sentence_col_name_list[0]
            assert self.sent1 in datasets.features, f"Can't find the name of col name of {self.sent1}."
            if len(sentence_col_name_list) != 1:
                self.sent2 = sentence_col_name_list[1]
                assert self.sent2 in datasets.features, f"Can't find the name of col name of {self.sent2}."
            self.label_enumerate_value = label_enumerate_value.replace(" ", "").split(",")
            self.num_label = len(self.label_enumerate_value)
            self.label_mapping = {label: i for i, label in enumerate(self.label_enumerate_value)}
            
    def process(self):
        encoded_dataset = self.datasets.map(self.convert_sentence_to_features, batched=True)
        colums = [colum for colum in encoded_dataset.column_names if colum not in self.datasets.column_names]
        encoded_dataset.set_format(type='torch', columns=colums)
        return encoded_dataset
    
    def convert_sentence_to_features(self, datasets):
        if self.sent2 is not None:
            encoding =  self.tokenizer(datasets[self.sent1],
                                  datasets[self.sent2],
                                  truncation=True, 
                                  padding='max_length',
                                  max_length=self.max_seq_length)
        else:
            encoding = self.tokenizer(datasets[self.sent1],
                            truncation=True, 
                            padding='max_length',
                            max_length=self.max_seq_length)
    
        # Adding extra data process
        if self.data_sourse == "hub":
            encoding["label_ids"] = datasets[self.label_col_name]
        elif self.data_sourse == "user":
            encoding["label_ids"] = [self.label_mapping[str(label)] for label in datasets[self.label_col_name]]
        return encoding

def load_dataset(path, name=None, data_files=None):
    # Local Data
    support_data_format = ["json", "csv", "text", "parquet"]
    if data_files is not None and path in support_data_format:
        return hf_load_dataset(path, data_files=data_files)
    
    # Huggingface data and Clue data
    datahub_base_dir = os.path.join(os.environ["HOME"], ".easynlp", "datahub")
    if not io.isdir(datahub_base_dir):
            io.makedirs(datahub_base_dir)
    assert io.isdir(datahub_base_dir), "%s is not a existing directory" % datahub_base_dir
    data_script_dir = os.path.join(datahub_base_dir, path)
    if not io.isdir(data_script_dir):
            io.makedirs(data_script_dir)
    if not io.exists(os.path.join(data_script_dir, f"{path}.py")):
        remote_root = "https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/script/"
        remote_url = os.path.join(remote_root, path, f"{path}.py")
        os.system("wget " + remote_url + " -P " + get_dir_name(data_script_dir))
    assert io.exists(os.path.join(data_script_dir, f"{path}.py"))
    data = hf_load_dataset(data_script_dir, name)
    return data
