import os
from os.path import exists, join
import json

def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_data_path(mode, encoder, data_path):
    paths = {}
    if mode == 'train':
        paths['train'] = os.path.join(data_path, '2000_efl_train_CNNDM_' + encoder + '.jsonl')
        paths['val']   = os.path.join(data_path, '2000_efl_test_CNNDM_' + encoder + '.jsonl')
    else:
        paths['test']  = os.path.join(data_path, '2000_efl_test_CNNDM_' + encoder + '.jsonl')
    return paths

def get_result_path(save_path, cur_model):
    result_path = join(save_path, '../result')
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)
    if not exists(model_path):
        os.makedirs(model_path)
    dec_path = join(model_path, 'dec')
    ref_path = join(model_path, 'ref')
    if not os.path.exists(dec_path):
        os.makedirs(dec_path)
    if not os.path.exists(ref_path):
        os.makedirs(ref_path)
    return dec_path, ref_path
