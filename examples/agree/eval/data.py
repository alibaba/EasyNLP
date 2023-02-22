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
    def __init__(self, jsonl_filename, max_txt_length=77):
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
        
        self.max_txt_length = max_txt_length

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

class EvalPromptDataset(Dataset):
    def __init__(self, jsonl_filename, max_txt_length=77):
        assert os.path.exists(jsonl_filename), "The prompt datafile {} not exists!".format(jsonl_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.queries = []
        with open(jsonl_filename, "r") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                query_id = obj['query_id']
                query = obj['items']
                self.queries.append((query_id, query))
        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        
        self.prompt = "a photo containing {object}"
        self.max_txt_length = max_txt_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id, txt_objs = self.queries[idx]
        txt_objs_prompts = [self.prompt.format(object=txt_obj) for txt_obj in txt_objs]

        # print(txt_objs_prompts)

        prompt_len = len(txt_objs_prompts)
        if prompt_len < 10:
            for obj_idx in range(prompt_len, 10):
                txt_objs_prompts.append("")
        elif prompt_len > 10:
            txt_objs_prompts = txt_objs_prompts[:10]

        text_prompts = tokenize(txt_objs_prompts, context_length=self.max_txt_length)

        return query_id, text_prompts

class EvalMaskDataset(Dataset):
    def __init__(self, jsonl_filename, max_txt_length=32):
        assert os.path.exists(jsonl_filename), "The prompt datafile {} not exists!".format(jsonl_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.queries = []
        with open(jsonl_filename, "r") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                query_id = obj['query_id']
                query_text = obj["query_text"]
                query = obj['items']
                self.queries.append((query_id, query_text, query))
        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        
        self.max_txt_length = max_txt_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_id, query_text, txt_objs = self.queries[idx]
        txt_objs_masks = [query_text.replace(txt_obj, "[MASK]") for txt_obj in txt_objs]

        # print(txt_objs_masks)

        prompt_len = len(txt_objs_masks)
        if prompt_len < 10:
            for obj_idx in range(prompt_len, 10):
                txt_objs_masks.append("")
        elif prompt_len > 10:
            txt_objs_masks = txt_objs_masks[:10]

        text_masks = tokenize(txt_objs_masks, context_length=self.max_txt_length)

        return query_id, text_masks

def get_eval_txt_dataset(args, max_txt_length=24):
    input_filename = args.text_data
    dataset = EvalTxtDataset(
        input_filename,
        max_txt_length=max_txt_length)
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

def get_eval_prompt_dataset(args, max_txt_length=32):
    input_filename = args.prompt_data
    dataset = EvalPromptDataset(
        input_filename,
        max_txt_length=max_txt_length)
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

def get_eval_mask_dataset(args, max_txt_length=32):
    input_filename = args.prompt_data
    dataset = EvalMaskDataset(
        input_filename,
        max_txt_length=max_txt_length)
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