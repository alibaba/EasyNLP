import os
import logging
import json

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

from training.data import DataInfo

from clip.clip import tokenize

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

def get_eval_txt_dataset(args):
    input_filename = args.text_data
    dataset = EvalTxtDataset(
        input_filename)
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.text_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_eval_img_dataset(args):
    img_filename = args.image_data
    dataset = EvalImgDataset(
        img_filename)
    num_samples = len(dataset)
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.img_batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)