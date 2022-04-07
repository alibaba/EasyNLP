#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import time
import torch
import common_io
from torch.utils.data import Dataset


import argparse
parser = argparse.ArgumentParser(description='PyTorch table IO')
parser.add_argument('--tables', default="", type=str, help='ODPS input table names')
parser.add_argument('--outputs', default="", type=str, help='ODPS output table names')
args = parser.parse_args()

train_table = args.tables

class TableDataset(torch.utils.data.IterableDataset):
    def __init__(self, table_path, slice_id=0, slice_count=1):
        self.table_path = table_path
        reader = common_io.table.TableReader(table_path,
                                             slice_id=slice_id,
                                             slice_count=slice_count,
                                             num_threads=0)
        self.row_count = reader.get_row_count()
        self.start_pos = reader.start_pos
        self.end_pos = reader.end_pos
        reader.close()
        super(TableDataset, self).__init__()
        print("table total_row_count:{}, start_pos:{}, end_pos:{}".format(self.row_count, self.start_pos, self.end_pos))
        self.cnt = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        print("worker_id:{}, num_workers:{}".format(worker_id, num_workers))

        table_start, table_end = self._get_slice_range(self.row_count, worker_id, num_workers, self.start_pos)
        table_path = "{}?start={}&end={}".format(self.table_path, table_start, table_end)
        print("table_path:%s" % table_path)

        def table_data_iterator():
            reader = common_io.table.TableReader(table_path, num_threads=1, capacity=1024)
            while True:
                try:
                    data = reader.read(num_records=1, allow_smaller_final_batch=True)
                    self.cnt += 1

                    if self.cnt % 50 == 0:
                        worker_info = torch.utils.data.get_worker_info()
                        print(worker_info)
                        print(f'processed {self.cnt} rows.')
                except common_io.exception.OutOfRangeException:
                    reader.close()
                    print(f'Finished one round. processed {self.cnt} rows.')
                    break
                yield data
        
        return table_data_iterator()

    def _get_slice_range(self, row_count, worker_id, num_workers, baseline=0):
        # div-mod split, each slice data count max diff 1
        size = int(row_count / num_workers)
        split_point = row_count % num_workers
        if worker_id < split_point:
            start = worker_id * (size + 1) + baseline
            end = start + (size + 1)
        else:
            start = split_point * (size + 1) + (worker_id - split_point) * size + baseline
            end = start + size
        return start, end


slice_id = int(os.environ.get('RANK', 0))
slice_count = int(os.environ.get('WORLD_SIZE', 1))

train_dataset = TableDataset(train_table, slice_id, slice_count)
_train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=False,
    pin_memory=False,
    sampler=None,
    num_workers=5,
    collate_fn=lambda x: x )

# for data in train_ld:
#     print(data)

for _epoch in range(10):
    for _step, batch in enumerate(_train_loader):
        if _step % 50 == 0:
            print(f"proocessed {_step} steps.")

    print(f'epoch {_epoch} finished.')
