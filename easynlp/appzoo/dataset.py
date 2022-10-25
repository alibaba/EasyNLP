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
import urllib

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import datasets
from datasets import list_datasets as hf_list_datasets
from datasets import load_dataset as hf_load_dataset

from ..utils import io, parse_row_by_schema, get_dir_name
from ..utils.logger import logger
from ..modelzoo import AutoTokenizer
from ..utils import EASYNLP_LOCAL_DATAHUB, EASYNLP_REMOTE_ROOT


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

            row = "\t".join([t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in row[0]])
        elif self.data_source in ["tar"]:
            row = self.data_rows[item]
        else:
            raise NotImplementedError

        if self.output_format == 'dict' and self.input_schema:
            row = parse_row_by_schema(row, self.input_schema)
        try:
            return self.convert_single_row_to_example(row)
        except NotImplementedError:
            return row
        except :
            logger.info("Failed row: {}".format(row))
            raise RuntimeError

    def __len__(self):
        if self.data_source in ["local", "oss", "tar"]:
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
        print(f'****{data_file}')
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


class GeneralDataset(BaseDataset):
    """ 
    Dataset is implemented for input data from 'load_dataset'.

    The default setting of 'GeneralDataset' is implemented for SequenceClassification, 
        so you need to choose the correct 'convert_single_row_to_example' and 'batch_fn' base on your application.

    In some special cases, you need to override the '__init__' function.

    Args:
        pretrained_model_name_or_path: for init tokenizer.
        data_file: input data file from 'load_dataset'
        max_seq_length: max sequence length of each input instance.
        
    """
    def __init__(self, data_file, pretrained_model_name_or_path:str, max_seq_length:int):    
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.max_seq_length = max_seq_length
        self.first_sequence = self.second_sequence = self.label_name = self.num_label = self.label_enumerate_value \
            = self.label_mapping = self.data_rows = None
        # Required by BaseDataset
        self.data_source = "local"
        assert isinstance(data_file, datasets.arrow_dataset.Dataset), \
            "The inputs data format must be datasets.arrow_dataset.Dataset from load_dataset()."
        dataset_info = getattr(data_file, "info", None)
        data_features = dataset_info.features
        self.column_names = list(data_file.features.keys())
        self.data_rows = [data_file[i] for i in range(data_file.num_rows)]
        
        if "ner_tags" in self.column_names:
            self.first_sequence = self.column_names[1]
            self.label_name = "ner_tags"
            if hasattr(data_features[self.label_name], 'num_classes'):
                self.num_label = data_features[self.label_name].num_classes
                self._label_enumerate_values = data_features[self.label_name].names
            elif hasattr(data_features[self.label_name], 'feature') and \
                hasattr(data_features[self.label_name].feature, 'num_classes'):
                self.num_label = data_features[self.label_name].feature.num_classes
                self._label_enumerate_values = data_features[self.label_name].feature.names
            else:
                raise RuntimeError("Can't auto inference the label, \
                            please check your 'ner_tags' in your dataset")
        else:
            self.first_sequence = self.column_names[0]
            if self.column_names[1] != "label":
                self.second_sequence = self.column_names[1]
            self.label_name = "label"
            self.num_label = data_features[self.label_name].num_classes
            self._label_enumerate_values = data_features[self.label_name].names
        self.label_map = {label: i for i, label in enumerate(self.label_enumerate_values)}

    def __len__(self):
        return len(self.data_rows)
    
    def __getitem__(self, item):
        row = self.data_rows[item]
        return self.convert_single_row_to_example(row)

    def __del__(self):
        pass
            
    @property
    def label_enumerate_values(self):
        """
            Returns the label enumerate values.
        """
        return self._label_enumerate_values

    def convert_single_row_to_example(self, row):

        text_a = row[self.first_sequence]
        text_b = row[self.second_sequence] if self.second_sequence else None
        label = row[self.label_name] if self.label_name else None

        encoding = self.tokenizer(text_a,
                                  text_b,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_length)
        if label in self.label_map.values():
            encoding['label_ids'] = label
        else:
            encoding['label_ids'] = self.label_map[label]
        return encoding

    def batch_fn(self, features):
        """
            Divide examples into batches.
        """
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}

def load_dataset(path, name=None, data_files=None):
    # Local Data
    support_data_format = ["json", "csv", "text", "parquet"]
    if data_files is not None and path in support_data_format:
        return hf_load_dataset(path, data_files=data_files)

    datahub_base_dir = EASYNLP_LOCAL_DATAHUB
    if not io.isdir(datahub_base_dir):
        io.makedirs(datahub_base_dir)
    assert io.isdir(datahub_base_dir), "%s is not a existing directory" % datahub_base_dir

    data_script_dir = os.path.join(datahub_base_dir, path)
    if not io.isdir(data_script_dir):
            io.makedirs(data_script_dir)
    if not io.exists(os.path.join(data_script_dir, f"{path}.py")):
        # Loading Huggingface Datasets list
        hug_datasets_list = hf_list_datasets()
        remote_base = EASYNLP_REMOTE_ROOT
        if path in hug_datasets_list or f"{path}/{name}" in hug_datasets_list:
            remote_root = os.path.join(remote_base, "script")
        else:
            remote_root = os.path.join(remote_base, "easynlp_script")
        remote_url = os.path.join(remote_root, path, f"{path}.py")
        try:
            os.system("wget " + remote_url + " -P " + get_dir_name(data_script_dir))
        except:
            raise RuntimeError
    assert io.exists(os.path.join(data_script_dir, f"{path}.py"))
    data = hf_load_dataset(data_script_dir, name)
    return data

def list_datasets():
    remote_base = EASYNLP_REMOTE_ROOT
    datahub_base_dir = EASYNLP_LOCAL_DATAHUB
    remote_url = os.path.join(remote_base, "easynlp_script", "datasets_list.txt")
    try:
        # os.system("wget " + remote_url + " -P " + get_dir_name(datahub_base_dir))
        urllib.request.urlretrieve(remote_url, os.path.join(datahub_base_dir, "datasets_list.txt"))
    except:
        raise RuntimeError
    local_file = os.path.join(datahub_base_dir, "datasets_list.txt")
    assert os.path.isfile(local_file)
    with open(local_file, "r") as f:
        file_stream = f.readlines()
    datasets_list = [data_name.strip() for data_name in file_stream]
    return datasets_list + hf_list_datasets()
    