#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mask_data.py
@Time    :   2022/07/02 16:49:50
@Author  :   jwdong 
@Version :   1.0
@Contact :   jobecini@gmail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import os
import json
import numpy as np
import logging
import sys

from typing import Dict, Tuple, Union, List
from tqdm import tqdm
from multiprocessing import cpu_count, Pool, Lock

def get_logger():
    logger = logging.getLogger('knowledge insert')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()
W_LOCK = Lock()
Knowledge_G: Dict[str, Dict[str, str]] = None
All_examples = 0
Total_examples = 0
Total_ner_examples = {_:0 for _ in range(1000)}

dep_marker_attrs={
    'SBV': [0],
    'DBL': [1],
    'ATT': [1],
    'ADV': [1],
    'COO': [0, 1],
    'LAD': [1],
    'RAD': [0],
    'HED': [0, 1]
}

sdp_marker_attrs={
    'AGT': [0, 1],
    'EXP': [0, 1],
}


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

def read_knowledge_txt(path: str) -> Dict[str, Dict[str, str]]:
    """get knowledge graph

    Args:
        path (str): the knowledge data path

    Returns:
        Dict[str, Dict[str, str]]: the knowledge data
    """
    triples = {}
    logger.info('start to read knowledge...')
    with open(path, 'r') as f:
        for line in f:
            try:
                ner_1, relationship, ner_2 = line.strip().split('\t')
                if ner_1 in triples:
                    triples[ner_1][relationship] = ner_2
                else:
                    triples[ner_1] = {relationship: ner_2}
            except ValueError as e:
                # print(e)
                ...
    logger.info('read knowledge over...')
   
    return triples

def get_positive_and_negative_examples(
        ner_data: str,
        negative_level: int = 3) -> Union[bool, Dict[str, List[str]]]:
    """get the positive examples and negative examples for the ner data

    Args:
        ner_data (str): the ner entity
        negative_level (int, optional): the deepth of the relationship. Defaults to 3.

    Returns:
        Union[bool, Dict[str, List[str]]]: if the `ner_data` not in `konwledge`, return False, otherwise, return the positive and negative examples
    """
    global Knowledge_G
    knowledge: Dict[str, Dict[str, str]] = Knowledge_G
    common_used = set()
    def get_data(key: str, 
                 data: Dict[str, str], 
                 results: List[str], 
                 deep: int, 
                 insert_flag: bool = False):
        """get the negative examples recursively

        Args:
            key (str): the ner
            data (Dict[str, str]): the related data about `key`
            results (List[str]): a list used to save the negative examples
            deep (int): the recursive number
            insert_flag (bool, optional): whether insert data to `results`. Defaults to False.
        """
        nonlocal knowledge
        # Avoid data interference between different generations, such as: 汤恩伯：三民主义;国民党; 国民党:三民主义 二阶和一阶数据重复了
        common_used.add(key)
        if deep == 0:
            return
        else:
            for key_item in data:
                if data[key_item] not in common_used and insert_flag == True:
                    results.append(data[key_item])
                if data[key_item] in knowledge and data[key_item] not in common_used:
                    get_data(data[key_item], knowledge[data[key_item]], results, deep - 1, True)
    
    all_examples = {
        'ner': ner_data,
        'positive_examples': [],
        'negative_examples': []
    }
    
    if ner_data in knowledge:
        tp_data = knowledge[ner_data]
        negative_examples = []
        if '描述' in tp_data:
            positive_example = tp_data['描述']
        else:
            keys = list(tp_data.keys())
            choice = np.random.choice([_ for _ in range(len(keys))], 1)[0]
            positive_example = tp_data[keys[choice]]
        # # the description usually contains the ner entity, if not, concate the `ner_data` and the positive example
        if ner_data in positive_example:
            all_examples['positive_examples'].append(positive_example)
        else:
            all_examples['positive_examples'].append(ner_data + positive_example)
        
        get_data(ner_data, tp_data, negative_examples, negative_level)
        # concate the ner entity and each negative example
        negative_examples = list(map(lambda x: ner_data + x if ner_data not in x else x, negative_examples))
        all_examples['negative_examples'] = negative_examples
        return all_examples
        
    return False

def preprocess(data: str, 
               total_prob: float = 0.15, 
               max_length: int = 512, 
               probs_each_: tuple = (0.3, 0.3, 0.4),
               deepth: int = 3, 
               seed: int = 42) -> List[List[Union[int, str]]]:
    """preprocess the text with the specific mask strategy(sdp mask, dep mask and random mask), 

    Args:
        data (str): the data to be preprocessed
        total_prob (float, optional): the probability of mask for the `data`. Defaults to 0.15.
        max_length (int, optional): maximum length of intercepted text. Defaults to 128.
        probs_each_ (tuple, optional): the probabilties of each three mask strategy (random, dep and sdp). Defaults to (0.3, 0.3, 0.4).
        deepth (int, optional): the deepth of the relationship. Defaults to 3.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        List[int]: the mask labels. e.g., [0, 1, 1, 0, 0, ..., 0]
    """
    np.random.seed(seed)
    dict_data = json.loads(data)

    dict_data['text'] = dict_data['text'][0][:max_length]
    text = dict_data['text']
    length = len(dict_data['text'])
    mask_labels = np.zeros(length)    
    
    mask_number = int(length * total_prob)
    rand_number, dep_number, sdp_number = map(lambda x: int(x * mask_number), probs_each_)
    
    # use seg_ids to record the start index of each segment in seg arr
    sep_ids = [0]
    counter = 0
    seg_data = dict_data['seg'][:max_length]
    for seg in seg_data[:-1]:
        tp_length = len(seg)
        sep_ids.append(tp_length+counter)
        counter += tp_length
    
    def check_ids(ids: Union[List[int], Tuple[int]]) -> bool:
        """check whether the id is reasonable (the location of 
        the id is occupied or the the index of id is bigger than the `max_length`)

        Args:
            ids (Union[List[int], Tuple[int]]): the ids list

        Returns:
            bool: if the `ids` is reasonable
        """
        nonlocal max_length
        for id in ids:
            if id >= max_length:
                return False
            if mask_labels[id] == 1:
                return False
        return True
    
    rand_selected_numbers = set()
    def random_mask():
        """mask the `mask_labels` according to the `rand_number` randomly
        """
        # random process
        nonlocal rand_number
        zeros = np.where(mask_labels == 0)[0]
        zero_arrs = np.array_split(zeros, rand_number)
        zero_choices = []
        for arr_ in zero_arrs:
            zero_choices.append(np.random.choice(arr_, 1))
        zero_choices = np.array(zero_choices)
        mask_labels[zero_choices] = 1
        ...
    
    def random_mask_old():
        """mask the `mask_labels` according to the `rand_number` randomly
        """
        # random process
        nonlocal rand_number
        zeros = np.where(mask_labels == 0)[0]
        
        zero_choices = np.random.choice(zeros, rand_number, replace=False)
        mask_labels[zero_choices] = 1
        print(mask_labels)
        quit()
    
    def seg2id(segs:Union[List[int], Tuple[int]]) -> List[List[int]]:
        """get the true id for each segment in `segs` based on `sep_ids`(contain the start index of each seg data)

        Args:
            segs (Union[List[int], Tuple[int]]): the segment id list

        Returns:
            List[List[int]]: the true ids for seg data
        """
        ids = []
        for seg_id in segs:
            length = len(seg_data[seg_id])
            start = sep_ids[seg_id]
            # for _ in range(start, start+length):
            #     ids.append(_)
            ids.append([_ for _ in range(start, start+length)])
        return ids
    
    
     # dep process
    dep_data = dict_data['dep']
    dep_data_new = []
    for index, dep_item in enumerate(dep_data):
        if max(dep_item[:2]) < max_length:
            dep_data_new.append(dep_item.copy())
    
    dep_numbers = set(_ for _ in range(len(dep_data_new)))
    dep_selected_numbers = set()
    dep_markers = []
    detail_info = []
    
     # sdp process
    sdp_data = dict_data['sdp']
    sdp_data_new = []
    for index, sdp_item in enumerate(sdp_data):
        if max(sdp_item[:2]) < max_length:
            sdp_data_new.append(sdp_item.copy())
    sdp_numbers = set(_ for _ in range(len(sdp_data_new)))
    sdp_selected_numbers =set() 
    sdp_markers = []

    def dep_sdp_mask(left_numbers: List[int], 
                     data_new: List[List[Union[int, str]]], 
                     markers_: List[List[int]], 
                     selected_numbers_: set, 
                     number_: int,
                     marker_attrs: Dict[str, List[int]]) -> int:
        """ mask the `mask_labels` for sdp and dep and record the maskers for each mask item

        Args:
            left_numbers (List[int]): the options that have not been used
            data_new (List[List[Union[int, str]]]): preprocessed data for original dep and sdp
            markers_ (List[List[int]]): a list that is uesd to save the maskers for each mask item
            selected_numbers_ (set): a set that is used to save the selected options
            number_ (int): the number of mask labels
            marker_attrs Dict[str, List[int]]: marker attributes

        Returns:
            int: 0 mean no mask, the others mean the number of masked ids
        """
        np.random.shuffle(left_numbers)
        for item_ in left_numbers:
            target_item = data_new[item_]
            seg_ids = np.array(target_item[:2]) - 1
            delete_ids = np.where(seg_ids < 1)[0]
            seg_ids = np.delete(seg_ids, delete_ids)
            temp_ids = seg2id(seg_ids)
            ids = []
            for item in temp_ids:
                ids += item.copy()
            if check_ids(ids):
                length_ = len(ids)
                if number_ > length_:
                    for id_ in ids:
                        mask_labels[id_] = 1
                    if target_item[2] in marker_attrs:
                        
                        detail_info.append([
                            target_item,
                            [seg_data[target_item[0] - 1],seg_data[target_item[1] - 1]],
                        ])
                        if len(temp_ids) == 1:
                            markers_.append([temp_ids[0][0], temp_ids[0][-1]])
                        elif len(temp_ids) == 2:
                            for i in marker_attrs[target_item[2]]:
                                markers_.append([temp_ids[i][0], temp_ids[i][-1]])
        
                    selected_numbers_.add(item_)
                    
                    return length_
                else:
                    return 0
        
        return 0
    # print(dep_number, sdp_number, rand_number)
    while (dep_number + sdp_number) > 0:
        choice_items = []
        choice_probs = []
        for index, value in enumerate([dep_number, sdp_number]):
            if value > 0:
                choice_items.append(index + 1)
                choice_probs.append(probs_each_[index])
        choice_probs = np.array(choice_probs) / sum(choice_probs)
        
        choice = np.random.choice(choice_items, size=1, p=choice_probs)[0]
        if choice == 1:
            left_numbers = list(dep_numbers - dep_selected_numbers)
            number = dep_sdp_mask(left_numbers, dep_data_new, dep_markers, dep_selected_numbers, dep_number, dep_marker_attrs)
            if number == 0:
                rand_number += dep_number
                dep_number = 0
            else:
                dep_number -= number
        else:
            left_numbers = list(sdp_numbers - sdp_selected_numbers)
            number = dep_sdp_mask(left_numbers, sdp_data_new, sdp_markers, sdp_selected_numbers, sdp_number, sdp_marker_attrs)
            if number == 0:
                rand_number += sdp_number
                sdp_number = 0
            else:
                sdp_number -= number
    
    text = list(text)
    mask_labels = mask_labels.tolist()
    for index in range(len(dep_markers)):
        dep_markers[index] += ['dep']
    for index in range(len(sdp_markers)):
        sdp_markers[index] += ['sdp']
    
    markers = dep_markers + sdp_markers
    new_markers = []
    for marker in markers:
        new_markers.append([marker[0], 0, marker[-1]])
        new_markers.append([marker[1], 1, marker[-1]])
    new_markers = sorted(new_markers, reverse=True)
    counter = 0
    while counter < len(new_markers):
        new_marker = new_markers[counter]
        insert_index = new_marker[0]
        if new_marker[1] == 1:
            insert_index += 1
        text.insert(insert_index, f"[{new_marker[-1]}]")
        mask_labels.insert(insert_index, 1)
        counter += 1
    
    text.insert(0, '[CLS]')
    mask_labels.insert(0, 0)
    detail_info.append(''.join(text.copy()))
    mask_labels = list(map(int, mask_labels))
    text = text[:max_length - 1] + ['[SEP]']
    mask_labels = mask_labels[:max_length - 1] + [0]
   
    # ner deal process
    ner_data = dict_data['ner']
    ner_data_str = []
    for ner_ in ner_data:
        tp_index = ner_[-1]
        if tp_index < max_length:
            ner_data_str.append(seg_data[tp_index])
        
    text_copy = ''.join(text).replace('[sdp]', '')
    text_copy = text_copy.replace('[dep]', '')
    
    return [text.copy(), mask_labels.copy(), ner_data_str]

def text_read(source_path: str, 
              save_path: str, 
              start: int, 
              end: int, 
              thread_id: int, 
              write_number: int = 1000):
    
    global MAX_LENGTH, SDP_PRO, DEP_PRO, RAND_PRO
    data_arr = []
    with open(source_path, 'r') as f:
        for i in tqdm(range(0, end), desc=f'Processing thread:{thread_id}'):
            line = f.readline()
            if i < start:
                continue
            
            data = preprocess(line, max_length=MAX_LENGTH, probs_each_=(RAND_PRO, DEP_PRO, SDP_PRO))
            data_arr.append(data)
            if i % write_number == 0 or i == (end - 1):
                W_LOCK.acquire()                    
                with open(save_path, "a") as f_:
                    for item in data_arr:
                        f_.write(str(item) + '\n')
                data_arr = []
                W_LOCK.release()

def main(thread_number: int, source_path: str, save_path: str, write_number: int = 1000):
    logger.info('start to counter data...')
    length = text_line_counter(source_path)
    logger.info(f'counter data over({length})~')
    split_gap = length // thread_number
    logger.info(f"total data number: {length}")
    logger.info(f'create thread pooling({thread_number} threads)...')
    pool = Pool(thread_number)
    pool_results = []
    logger.info('pooling created~')
    logger.info(f'each thread process data number:{split_gap}')
    logger.info('start threads...')
    for i in range(thread_number):
        start = i * split_gap
        end = (i + 1) * split_gap if i < (thread_number - 1) else length
        logger.info(f"thread {i} is starting[{start}/{end}]")
        pool_results.append(pool.apply_async(text_read, (source_path, save_path, start, end, i, write_number)))
    
    pool.close()
    pool.join()
if __name__ == '__main__':
    # for convenience, use sys to pass parameters
    path = sys.argv[1]
    save_path = sys.argv[2]
    if os.path.exists(save_path):
        os.remove(save_path)
    MAX_LENGTH = int(sys.argv[3])
    SDP_PRO = float(sys.argv[4])
    DEP_PRO = float(sys.argv[5])
    RAND_PRO = float(sys.argv[6])

    main(48, path, save_path, 100)