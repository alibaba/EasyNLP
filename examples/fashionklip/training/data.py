import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb
import json

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets

from clip.clip import tokenize

import io
from PIL import Image
import random

from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 933120000

def parse_obj_dict(file_path, keywords):
    ids_objs = {}
    with open(file_path, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            item_id = obj[keywords[0]]
            item_objs = obj[keywords[1]]
            
            ids_objs[item_id] = item_objs
    
    return ids_objs

class JsonlDataset(Dataset):
    def __init__(self, jsonl_filename, img_filename):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)
        assert os.path.exists(img_filename), "The image npz datafile {} not exists!".format(img_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.samples = []
        with open(jsonl_filename, "r") as fin:
            for line in fin:
                if len(line) >0:
                    try:
                        obj = json.loads(line.strip())
                        query_id = obj['query_id']
                        query = obj['query_text']
                        for target in obj['item_ids']:
                            self.samples.append((query_id, query, target))
                    except:
                        print('-------------------')
                        print(line)
                        print('-------------------')

        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        print((f'Finished loading jsonl data from {jsonl_filename}.'))
        
        logging.debug(f'Loading image npzfile from {img_filename}.')
        print(f'Start loading image npzfile from {img_filename}.')

        self.imgs = np.load(img_filename, "r")

        logging.debug(f'Finished loading image npzfile from {img_filename}.')
        print(f'Finished loading image npzfile from {img_filename}.')

    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query_id, query, img_id = self.samples[idx]
        image = self._read_img_tensor_from_npzfile(img_id)
        text = tokenize([str(query)])[0]
        return image, text

class Jsonl_Concept_Dataset_Update(Dataset):
    def __init__(self, jsonl_filename, img_filename, txt_id_filename):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)
        assert os.path.exists(img_filename), "The image npz datafile {} not exists!".format(img_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.samples = []

        self.text2id = parse_obj_dict(txt_id_filename, ["phrase", "query_id"])
        self.pred_keys = self.text2id.keys()

        with open(jsonl_filename, "r") as fin:
            for line in fin:
                if len(line) > 0:
                    try:
                        obj = json.loads(line.strip())
                        query_id = obj['query_id']
                        query = obj['query_text']
                        phrases = obj['match_phrases']
                        phrases = [phrase_item for phrase_item in phrases if len(phrase_item) >= 2 and phrase_item in self.pred_keys]
                        # phrases = [phrase_item for phrase_item in phrases if len(phrase_item) >= 3]
                        for target in obj['item_ids']:
                            self.samples.append((query_id, query, target, phrases))
                    except:
                        print('-------------------')
                        print(line)
                        print('-------------------')

        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        print((f'Finished loading jsonl data from {jsonl_filename}.'))
        
        logging.debug(f'Loading image npzfile from {img_filename}.')
        print(f'Start loading image npzfile from {img_filename}.')

        self.imgs = np.load(img_filename, "r")
        self.prompt = "a photo of {}"

        logging.debug(f'Finished loading image npzfile from {img_filename}.')
        print(f'Finished loading image npzfile from {img_filename}.')

        self.con_len = 5

    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query_id, query, img_id, query_concept = self.samples[idx]
        image = self._read_img_tensor_from_npzfile(img_id)
        text = tokenize([str(query)])[0]

        concept_len = len(query_concept)
        if concept_len < self.con_len:
            for obj_idx in range(concept_len, self.con_len):
                query_concept.append("")
        elif concept_len >= self.con_len:
            query_concept = query_concept[:self.con_len]
            
        concepts = tokenize([self.prompt.format(query_item) for query_item in query_concept])
        concept_ids = [self.text2id[item] if item != "" else -1 for item in query_concept]

        return image, text, concepts, concept_ids

class Jsonl_All_Concept_Dataset_Update_Hierarchy(Dataset):
    def __init__(self, jsonl_filename, img_filename, txt_id_filename, kb_txt_id_filename):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)
        assert os.path.exists(img_filename), "The image npz datafile {} not exists!".format(img_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.samples = []

        # self.predictions = parse_obj_dict(txt_img_prediction, ["query_id", "item_ids"])
        self.text2id = parse_obj_dict(txt_id_filename, ["phrase", "query_id"])
        self.text2id_kb = parse_obj_dict(kb_txt_id_filename, ["phrase", "query_id"])

        self.concept_fathers = parse_obj_dict("./tmp/fashion_kb/icbu_concepts_fathers.jsonl", ["phrase", "phrase_father"])
        self.kb_concept_fathers = parse_obj_dict("./tmp/fashiongen/train/fashion-gen_concepts_fathers.jsonl", ["phrase", "phrase_father"])

        self.pred_keys = self.text2id.keys()
        self.pred_kb_keys = self.text2id_kb.keys()

        with open(jsonl_filename, "r") as fin:
            for line in fin:
                if len(line) > 0:
                    try:
                        obj = json.loads(line.strip())
                        query_id = obj['query_id']
                        query = obj['query_text']
                        phrases = obj['phrases']
                        match_phrases = obj['match_phrases']

                        phrases = [phrase_item for phrase_item in phrases if 2 <= len(phrase_item.split()) <= 6 and phrase_item in self.pred_keys and phrase_item not in match_phrases]
                        match_phrases = [phrase_item for phrase_item in match_phrases if len(phrase_item) >= 2 and phrase_item in self.pred_kb_keys]
                        for target in obj['item_ids']:
                            self.samples.append((query_id, query, target, phrases, match_phrases))
                    except:
                        print('-------------------')
                        print(line)
                        print('-------------------')

        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        print((f'Finished loading jsonl data from {jsonl_filename}.'))
        
        logging.debug(f'Loading image npzfile from {img_filename}.')
        print(f'Start loading image npzfile from {img_filename}.')

        self.imgs = np.load(img_filename, "r")
        self.prompt = "a photo of {}"

        logging.debug(f'Finished loading image npzfile from {img_filename}.')
        print(f'Finished loading image npzfile from {img_filename}.')

        self.con_len = 5

    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.samples)
    
    def _process_concepts(self, concepts, concept_father_dict, text2id_dict):
        concept_len = len(concepts)
        concept_fathers = list(set([concept_father_dict[concept_item] for concept_item in concepts if concept_item in concept_father_dict and concept_father_dict[concept_item] != "" ]))
        concept_grand_fathers = list(set([concept_father_dict[concept_item] for concept_item in concept_fathers if concept_item in concept_father_dict and concept_father_dict[concept_item] != ""]))

        father_len = len(concept_fathers)
        grand_father_len = len(concept_grand_fathers)

        if concept_len < self.con_len:
            if (concept_len + father_len) < self.con_len:
                concepts.extend(concept_fathers)
                concepts = list(set(concepts))
                concept_len = len(concepts)

                if (concept_len + grand_father_len) < self.con_len:
                    concepts.extend(concept_fathers)
                    concepts = list(set(concepts))
                    concept_len = len(concepts)
        
        if concept_len < self.con_len:
            for obj_idx in range(concept_len, self.con_len):
                concepts.append("")
        elif concept_len >= self.con_len:
            concepts = concepts[:self.con_len]
        
        tokenized_concepts = tokenize([self.prompt.format(concept_item) for concept_item in concepts])
        concept_ids = [text2id_dict[item] if item != "" and item in text2id_dict else -1 for item in concepts]

        return tokenized_concepts, concept_ids
    
    def __getitem__(self, idx):
        query_id, query, img_id, query_concept, query_kb_concept = self.samples[idx]
        image = self._read_img_tensor_from_npzfile(img_id)
        text = tokenize([str(query)])[0]

        data_concepts, data_concept_ids = self._process_concepts(query_concept, self.concept_fathers, self.text2id)
        kb_concepts, kb_concept_ids = self._process_concepts(query_kb_concept, self.kb_concept_fathers, self.text2id_kb)

        return image, text, data_concepts, kb_concepts, data_concept_ids, kb_concept_ids


class EvalImgDataset(Dataset):
    def __init__(self, img_filename):
        assert os.path.exists(img_filename), "The image npz datafile {} not exists!".format(img_filename)
        
        logging.debug(f'Loading image npzfile from {img_filename}.')
        self.imgs = np.load(img_filename, "r")
        self.img_ids = list(self.imgs.keys())
        logging.debug(f'Finished loading image npzfile from {img_filename}.')

    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self._read_img_tensor_from_npzfile(img_id)
        return img_id, image


class EvalImgDatasetList(Dataset):
    def __init__(self, img_file_sets):
        self.img_ids = []
        self.imgs = {}
        for img_filename in img_file_sets:
            assert os.path.exists(img_filename), "The image npz datafile {} not exists!".format(img_filename)
            logging.debug(f'Loading image npzfile from {img_filename}.')

            self.imgs.update(np.load(img_filename, "r"))
            self.img_ids.extend(list(self.imgs.keys()))

            logging.debug(f'Finished loading image npzfile from {img_filename}.')

    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self._read_img_tensor_from_npzfile(img_id)
        return img_id, image


class EvalTxtDataset(Dataset):
    def __init__(self, jsonl_filename):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.queries = []
        with open(jsonl_filename, "r") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                query_id = obj['query_id']
                query = obj['query_text']
                self.queries.append((query_id, query))
        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id, query = self.queries[idx]
        text = tokenize([str(query)])[0]
        return query_id, text


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    sizes = json.load(open(sizes_filename, 'r'))
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards

def get_jsonl_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    img_filename = args.train_img if is_train else args.val_img

    if is_train and args.is_concept and args.is_update:
        if args.is_data_concept:
            dataset = Jsonl_All_Concept_Dataset_Update_Hierarchy(
                input_filename,
                img_filename,
                args.txt_id_filename,
                args.kb_txt_id_filename)
        else:
            dataset = Jsonl_Concept_Dataset_Update(
                input_filename,
                img_filename,
                args.txt_img_prediction,
                args.txt_id_filename)
    else:
        dataset = JsonlDataset(
            input_filename,
            img_filename)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "jsonl":
        return get_jsonl_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['jsonl']:
            return get_json_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    return data


def get_eval_img_dataset(args):
    img_filename = args.img_data
    dataset = EvalImgDataset(
        img_filename)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_eval_img_dataset_list(args):
    img_file_sets = args.img_data_sets
    dataset = EvalImgDatasetList(
        img_file_sets)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_eval_txt_dataset(args, is_kb=False, max_txt_length=77):
    input_filename = args.concept_data if not is_kb else args.kb_concept_data
    dataset = EvalTxtDataset(
        input_filename,
        max_txt_length=max_txt_length)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
