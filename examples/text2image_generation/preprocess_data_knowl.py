# import sys
# sys.path.append('../')
# from load_dataset import *
# from load_path import *
# from V1.add_lattice import equip_chinese_ner_with_lexicon, equip_wukong_lexicon

import argparse
import collections
from fastNLP import DataSet, Instance
from requests import request
from tqdm import tqdm
from functools import partial

from easynlp.modelzoo import BertTokenizer
from easynlp.utils import get_pretrain_model_path

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        current = self.root
        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0
    
    def get_lexicon(self,chars):
        result = []
        for i in range(len(chars)):
            current = self.root
            for j in range(i, len(chars)):
                current = current.children.get(chars[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j, ''.join(chars[i:j+1])])

        return result

def load_dataset(path):
    print("load dataset")
    dataset = DataSet()
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            data = line.strip().split('\t')
            if len(data) == 2:
                idx, text = line.strip().split('\t')
                chars = tokenizer.tokenize(text)
                dataset.append(Instance(idx=idx, text=text, chars=chars))
            elif len(data) == 3:
                idx, text, imgbase64 = line.strip().split('\t')
                chars = tokenizer.tokenize(text)
                dataset.append(Instance(idx=idx, text=text, chars=chars, imgbase64=imgbase64))
    dataset.add_seq_len('chars',new_field_name='seq_len')
    return dataset


def load_cn_dbpedia_word_list(entity_path, drop_characters=True):
    w_list = []
    w_map = {}
    with open(entity_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines[1:]:
            data = line.strip().split('\t')
            w = data[0]
            w_list.append(w)
            w_map[w] = data[1]
    if drop_characters:
        w_list = list(filter(lambda x:len(x) != 1, w_list))
    
    return w_list, w_map


def equip_data_lexicon(datasets, w_list, w_map):
    print("equip_data_lexicon")
    def get_skip_path(chars, w_trie):
        result = w_trie.get_lexicon(chars)
        return result
    
    def get_skip_id(lexicons, w_map):
        lex_ids = []
        for lex in lexicons:
            idx = w_map[lex[2]]
            lex_ids.append(idx)
        return lex_ids

    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    # datasets.apply_field(get_chars, 'text', 'chars')
    datasets.apply_field(partial(get_skip_path,w_trie=w_trie), 'chars', 'lexicons')
    datasets.apply_field(partial(get_skip_id, w_map=w_map), 'lexicons', 'lex_ids')
    datasets.add_seq_len('lexicons','lex_num')
    datasets.apply_field(lambda x:list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
    datasets.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x:x[2],lexicons))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e
        return pos_e
    
    datasets.apply(concat,new_field_name='lattice')
    datasets.set_input('lattice')
    datasets.apply(get_pos_s,new_field_name='pos_s')
    datasets.apply(get_pos_e, new_field_name='pos_e')

    return datasets


def main(args):
    # load dataset
    dataset = load_dataset(args.input_file)

    # load cn_dbpedia words
    w_list, w_map = load_cn_dbpedia_word_list(args.entity_map_file) 

    # get lattice and entity position
    dataset = equip_data_lexicon(dataset, w_list, w_map)
    
    # save to file
    fout = open(args.output_file, 'w')
    for item in dataset:
        lex_ids = ' '.join(item['lex_ids'])
        pos_s = ' '.join([str(x) for x in item['pos_s']])
        pos_e = ' '.join([str(x) for x in item['pos_e']])
        if len(item['chars']) + len(item['lex_ids']) != len(item['pos_s']):
            print(item['chars'], item['lex_ids'], item['pos_s'])
        seq_len = item['seq_len']
        if 'imgbase64' in item:
            fout.write(item['idx']+'\t'+item['text']+'\t'+lex_ids+'\t'+pos_s+'\t'+pos_e+'\t'+str(seq_len)+'\t'+item['imgbase64']+'\n')
        else:
            fout.write(item['idx']+'\t'+item['text']+'\t'+lex_ids+'\t'+pos_s+'\t'+pos_e+'\t'+str(seq_len)+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--entity_map_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(get_pretrain_model_path('hfl/chinese-roberta-wwm-ext'))
    main(args)
