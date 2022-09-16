#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_threads.py
@Time    :   2022/06/28 15:22:11
@Author  :   jwdong 
@Version :   1.0
@Contact :   jobecini@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
# import linecache
import sys
import psutil
import os
import time
import json
import logging
import torch
import copy
import gc
import argparse


from math import *
from typing import Dict, List, Tuple, Union
from ltp import LTP
from operator import attrgetter, itemgetter
from collections import namedtuple
from multiprocessing import cpu_count, Pool, Lock


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


LTP_DATA = namedtuple('ltp_data', 'seg hidden pos ner srl dep sdp')

threads_setting = 48


LOCK = Lock()
def text_read_all(path: str) -> List[Dict[str, str]]:
    """read text

    Args:
        path (str): file path

    Returns:
        List[Dict[str, str]]: all data
    """
    with open(path, 'r') as f:
        data = f.readlines()
    data = list(map(eval, data))
    return data
    ...

def text_line_counter(path: str) -> int:
    """counter file

    Args:
        path (str): path

    Returns:
        int: number of lines in a file
    """
    with open(path, 'r') as f:
        counter = 0
        for _ in f:
            counter += 1
    
    return counter
    ...
def text_read(path: str, start:int, num: int) -> List[Dict[str, str]]:
    """read data from *start* line to *start + num* line in specific file.

    Args:
        path (str): file path
        start (int): start line
        num (int): number of data

    Returns:
        List[Dict[str, str]]: all data
    """
    with open(path, 'r') as f:
        counter = 0
        data_lines = []
        while counter < (start + num):
            if counter < start:
                f.readline()
            else:
                data_lines.append(eval(f.readline()))
            counter += 1
 
    return data_lines
    ...

def ltp_process(ltp: LTP, data: List[Dict[str, Union[str, List[Union[int, str]]]]]):
    """use ltp to process the data

    Args:
        Dict ([str, str]): data
        example:
            {'text':['我叫汤姆去拿伞。'],...}

    Returns:
        Dict[str, str]: result
    """
    
    new_data = list(map(lambda x:x['text'][0].replace(" ", ""), data))
      
    seg, hiddens = ltp.seg(new_data)
    result = {}
    result['seg'] = seg
    # data['pos'] = ltp.pos(hidden)
    result['ner'] = ltp.ner(hiddens)
    # data['srl'] = ltp.srl(hidden)
    result['dep'] = ltp.dep(hiddens)
    result['sdp'] = ltp.sdp(hiddens)
    for index in range(len(data)):
        data[index]['text'][0] = data[index]['text'][0].replace(" ", "")
        data[index]['seg'] = result['seg'][index]
        data[index]['ner'] = result['ner'][index]
        data[index]['dep'] = result['dep'][index]
        data[index]['sdp'] = result['sdp'][index]
    



def memory_info():
    """return memory usage
    """
    print(f"当前进程的内存使用:{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024} GB")

def json_str(data: Dict[str, str]) -> str:
    return json.dumps(data, ensure_ascii=False)
    
def preprocess(thread_number: int, path: str, ltp:LTP, save_path:str, indexs: Tuple[int], split_gap: int, start_number: int) -> str:
    """preprocess the data

    Args:
        thread_number (int): thread number
        path (str): data path
        ltp (LTP): an instance of ltp
        save_path (str): preprocessed data save path
        indexs (Tuple[int]): (start and end)
        split_gap (int): each *split_number* to save data

    Returns:
        str: result str
    """
    logging.info(f"thread {thread_number} run ~")
    length = indexs[-1] - indexs[0]
    start = indexs[0]
    split_number = ceil(length / split_gap)
    
    with open(path, 'r') as f:
        counter = 0
        data_lines = []
        gap_number = start_number + 1
        data_length = 0
        while counter < indexs[-1]:
            if counter < start:
                f.readline()
                counter += 1
                continue
            
            data_lines.append(eval(f.readline()))
            data_length += 1
            
            if data_length > start_number * split_gap:
                number = gap_number * split_gap if gap_number < split_number else length
                if data_length == number:
                    # print(counter, (gap_number - 1) * split_gap, number)
                    logging.info(f'thread {thread_number} the {gap_number} time start to processing data, the number is {len(data_lines)}...')
                    
                    ltp_process(ltp, data_lines)
                    
                    LOCK.acquire()
                    torch.cuda.empty_cache()
                
                    logging.info(f'thread {thread_number} the {gap_number} time start to write data, the number is {len(data_lines)}...')
                    data_lines = list(map(json_str, data_lines))
                    with open(save_path, 'a') as f_:
                        for index, _ in enumerate(data_lines):
                            f_.write(_ + '\n')
                    LOCK.release()
                    logging.info(f'thread {thread_number} the {gap_number} time over ~')
                    gap_number += 1
                    data_lines = []
                    gc.collect()   
            counter += 1
    logging.info(f'thread {thread_number} finished ~')

def main(path: str, save_path: str, cuda_number: int = 1, process_split: int = 100, start_number: int = 0):
    """preprocess the data

    Args:
        path (str): original file path
        save_path (str): save path
        process_split (int, optional): write to file each *process_split* numbers. Defaults to 100.
    """
    global THREADS
    
    logging.info('start to count data...')
    length = text_line_counter(path)
    logging.info(f'counter data over({length})~')
    split_gap = length // THREADS
    logging.info(f"total data number: {length}")
    logging.info(f'create thread pooling({THREADS} threads)...')
    pool = Pool(THREADS)
    pool_results = []
    logging.info('pooling created~')
    logging.info(f'each thread process data number:{split_gap}')
    
    logging.info('start threads...')
    for i in range(THREADS):
        cuda_ = i % cuda_number
        # print(cuda_)
        logging.info(f"thread {i} is assigned to gpu:{cuda_}")
        ltp = LTP(cuda_number=cuda_)
        start = i * split_gap
        end = (i + 1) * split_gap if i < (THREADS - 1) else length
        pool_results.append(pool.apply_async(preprocess, (i, path, ltp, save_path, (start, end), process_split, start_number)))

    pool.close()
    pool.join()
    # print(pool_results[0].get())
    # data = text_read(path, 10, data_num)
    ...
if __name__ == "__main__":
    # for convenience, use sys to pass parameters
    path = sys.argv[1]
    save = sys.argv[2]
    if os.path.exists(save):
        os.remove(save)
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
    # thread number
    threads_setting = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # threads_setting = 1
    THREADS = threads_setting if threads_setting != 0 else cpu_count()
    # 每个进程同时最多处理多少数据，32G显存最多220
    split_number = 50
    # 从那一份数据开始预处理
    start = 0
    # 调用的显卡数量
    gpu_number = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    main(path, save, gpu_number, split_number, start)