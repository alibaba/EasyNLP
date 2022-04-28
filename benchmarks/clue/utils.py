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
import time
import torch
import numpy as np
from typing import Union, List
from easynlp.appzoo.api import get_application_model, get_application_dataset, get_application_evaluator
from datasets import load_dataset as hf_load_dataset
from datasets import DatasetDict, Dataset
from easynlp.modelzoo import AutoConfig, AutoTokenizer
from benchmarks.clue.preprocess import tasks2processor, DatasetPreprocessor
from easynlp.appzoo.dataset import BaseDataset
from easynlp.utils import io, parse_row_by_schema
from easynlp.utils.logger import logger

from torch.utils.data import Dataset

class CLUEDataset(Dataset):
    def __init__(self,
                 dataset: Union[Dataset, DatasetDict],
                 preprocessor: DatasetPreprocessor,
                 set_name, # train / dev / test
                 input_schema,
                 label_name=None,
                 multi_label=False,
                 output_format="dict",
                 *args,
                 **kwargs):
        self.skip_first_line = False
        self.output_format = output_format
        self.data_source = "local"
        self.cnt = 0

        self.tokenizer = preprocessor.tokenizer
        self.max_seq_length = preprocessor.max_seq_length
        self.multi_label = multi_label
        self.data_rows = [i for i in dataset[set_name]]
        self.input_schema = input_schema
        self._label_enumerate_values = preprocessor.get_labels()
        self.max_num_labels = len(self._label_enumerate_values)
        self.first_sequence = None
        self.second_sequence = None
        self.label_name = label_name
        self.label_map = dict({value: idx for idx, value in enumerate(self._label_enumerate_values)})

    def print_examples(self):
        for data in self.data_rows[:5]:
            logger.info("*** Example ***")
            # logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in data['input_ids']]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in data['attention_mask']]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in data['token_type_ids']]))
            if 'label_ids' in data.keys():
                logger.info("label: (id = %d)" % (data['label_ids']))
            # logger.info("input length: %d" % (input_len))

    @property
    def label_enumerate_values(self):
        return self._label_enumerate_values

    def convert_single_row_to_example(self, row):
        pass

    def batch_fn(self, features):
        batch = {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}
        # print("label_ids=", batch["label_ids"])
        return batch


    def get_odps_reader(self, data_file, is_training, reader_buffer_size, kwargs=None):
        pass

    def get_odps_reader2(self, table_path, is_training, reader_buffer_size, kwargs=None):
        pass

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
        pass

    def __getitem__(self, item):
        row = self.prepare_row(item)
        return row


    def __len__(self):
        return len(self.data_rows)

    def prepare_row(self, item: int) -> str:
        row = self.data_rows[item]
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
        pass

    @property
    def label_enumerate_values(self):
        return ["0", "1"]

    @property
    def labels(self):
        labels = []
        for row in self.data_rows:
            # row = parse_row_by_schema(row, self.input_schema)
            label = row[self.label_name] if self.label_name else None
            labels.append(int(label))
        return np.asarray(labels)

    def convert_single_row_to_example(self, row):
        pass

    def get_odps_input_schema(self):
       pass



def download_data():
    '''
    default saved in the root directory of the project.
    '''
    data_base_dir = os.path.join(os.environ['HOME'], '.benchmark', 'clue')
    if not io.exists(data_base_dir):
        io.makedirs(data_base_dir)
    assert io.isdir(data_base_dir), '%s is not a existing directory' % data_base_dir
    if not io.exists(data_base_dir + 'CLUEdatasets1.1.tar.gz'):
        # Use the remote mapping file
        """with urllib.request.urlopen("https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/CLUEdatasets1.1.tar.gz") as f:

        model_name_mapping = json.loads(f.read().decode('utf-8'))
        """
        while True:
            try:
                if os.path.exists('CLUEdatasets1.1.tar.gz'):
                    break
                print('Trying downloading CLUEdatasets1.1.tar.gz')
                os.system(
                    'wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/clue/CLUEdatasets1.1.tar.gz'
                )
                os.system(
                    'tar -zxvf CLUEdatasets1.1.tar.gz'
                )
                print('Success')
            except Exception:
                time.sleep(2)



def load_dataset(
        preprocessor: DatasetPreprocessor,
        clue_name: str = 'clue',
        task_name: str = 'csl',
        is_training: bool = True,
        cache_dir: str = './tmp'
)-> CLUEDataset:
    download_data()
    if not os.path.exists:
        os.makedirs(cache_dir)
    data_dir = os.path.join('CLUEdatasets', "{}".format(task_name))
    data_files = {}
    if is_training:
        data_files["train"] = os.path.join(data_dir, "train.json")
        data_files["dev"] = os.path.join(data_dir, "dev.json")
    else:
        data_files["test"] = os.path.join(data_dir, "test.json")
    print('data_files=', data_files)
    extension = "json"
    # 获得huggingface dataset（待datahub开发完毕则转为datahub）
    datasets = hf_load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
    hf_datasets = datasets.map(
        preprocessor.convert_examples_to_features,
        batched=True,
        remove_columns=preprocessor.remove_column_list,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset line_by_line",
    )
    # 将huggingface dataset适配到easynlp的dataset
    easynlp_datasets = dict()
    for set_name in data_files.keys():
        easynlp_datasets[set_name] = CLUEDataset(
            dataset=hf_datasets,
            preprocessor=preprocessor,
            set_name=set_name,
            input_schema=hf_datasets[set_name].column_names,
            label_name=preprocessor.get_column_name()['label'][0] if is_training else None,
        )
        # print examples
        easynlp_datasets[set_name].print_examples()
    return easynlp_datasets
