# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 7:47 pm.
# @Author  : JianingWang
# @File    : datasets
from typing import Optional, List
from datasets import Dataset, DatasetInfo, NamedSplit
from datasets.table import Table, list_table_cache_files


class DatasetK(Dataset):
    def __init__(
            self,
            arrow_table: Table,
            info: Optional[DatasetInfo] = None,
            split: Optional[NamedSplit] = None,
            indices_table: Optional[Table] = None,
            fingerprint: Optional[str] = None,
    ):
        self.custom_cache_files = None
        super(DatasetK, self).__init__(arrow_table, info, split, indices_table, fingerprint)


    @property
    def cache_files(self) -> List[dict]:
        """The cache files containing the Apache Arrow table backing the dataset."""
        if self.custom_cache_files:
            return self.custom_cache_files
        cache_files = list_table_cache_files(self._data)
        if self._indices is not None:
            cache_files += list_table_cache_files(self._indices)
        return [{"filename": cache_filename} for cache_filename in cache_files]

    def set_cache_files(self, custom_cache_files):
        self.custom_cache_files = custom_cache_files
