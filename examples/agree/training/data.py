import os
import sys
import math
import logging
import functools
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
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets

from clip.clip import tokenize

import io
from PIL import Image,ImageFile
import random

from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

class JsonDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, split='train'):
        logging.debug(f'Loading json data from {input_filename}.')
        
        with open(input_filename,'r') as f:
            items = json.load(f)
        
        print(len(items))

        self.transforms = transforms
        self.samples = []
        self.data_root = "./tmp/datasets/flickr30k_images"

        if split == 'train':
            for item in tqdm(items):
                image_idx = item["image_id"]
                image_id = item[img_key]
                caption = item[caption_key]
                self.samples.append((image_idx, image_id, caption))
            
            img_filename = "annotation/flickr30k_train_images.npz"
        
        elif split == 'val':
            for index, item in tqdm(enumerate(items)):
                image_idx = index
                image_id = item[img_key]
                captions = item[caption_key]
                for caption in captions:
                    self.samples.append((image_idx, image_id, caption))

            img_filename = "annotation/flickr30k_val_images.npz"

        logging.debug(f'Loading image npzfile from {img_filename}.')
        self.imgs = np.load(os.path.join(self.data_root, img_filename), "r")
        
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.samples)
    
    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __getitem__(self, idx):
        img_idx, image, caption = self.samples[idx]

        images = self._read_img_tensor_from_npzfile(str(img_idx))
        texts = tokenize([str(caption)])[0]

        return images, texts


class Json_All_Dataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, split='train', is_object=False):
        logging.debug(f'Loading json data from {input_filename}.')

        self.transforms = transforms
        self.samples = []

        with open(input_filename, "r") as fin:
            for line in tqdm(fin):
                item = json.loads(line.strip())
                image_idx = item["image_id"]
                image_id = item[img_key]
                caption = item[caption_key]
                segs = item["segs"]
                seg_preds = item["preds"]
                seg_preds = [list(item_pred.items()) for item_pred in seg_preds if list(item_pred.items())[0][0] != "object"]
                self.samples.append((image_idx, image_id, caption, segs, seg_preds))
        
        # print(self.samples[0])
        self.data_root = "./tmp/datasets/flickr30k_images"

        img_filename = "annotation/flickr30k_train_images.npz"
        logging.debug(f'Loading image npzfile from {img_filename}.')

        self.imgs = np.load(os.path.join(self.data_root, img_filename), "r")
        
        self.img_obj_dict = {}
        for split_id in ["train", "test", "val"]:
            obj_file = "flickr30k_{}_imgs_objs.jsonl".format(split_id)
            img_obj_dict = parse_obj_dict(os.path.join(self.data_root, "objects", obj_file), ["image_id", "image_objects"])
            self.img_obj_dict.update(img_obj_dict)
        
        img_vg_filename = "./tmp/VG/VG_images.224.npz"
        img_mask_filename = "./tmp/VG/VG_mask_images.224.npz"

        logging.debug(f'Loading image npzfile from {img_vg_filename}.')
        self.vg_imgs = np.load(img_vg_filename, "r")
        logging.debug(f'Finished loading image npzfile from {img_vg_filename}.')

        logging.debug(f'Loading image npzfile from {img_mask_filename}.')
        self.vg_mask_imgs = np.load(img_mask_filename, "r")
        logging.debug(f'Finished loading image npzfile from {img_mask_filename}.')

        self.split = split
        self.img_mask_dict = parse_obj_dict("./tmp/VG/vocab_VG_images_objs.jsonl", ["vocab", "objects"])
        
        self.prompt = "a photo containing {object}"
        self.is_object = is_object

        img_vg_mask_filename = os.path.join(self.data_root, "flickr30k_visual_grounding.224.npz")
        
        logging.debug(f'Loading image npzfile from {img_vg_mask_filename}.')
        self.xvg_mask_imgs = np.load(img_vg_mask_filename, "r")
        logging.debug(f'Finished loading image npzfile from {img_vg_mask_filename}.')

        logging.debug('Done loading data.')
    
    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)
    
    def _read_vg_mask_img_tensor_from_npzfile(self, img_id):
        img_array = self.xvg_mask_imgs[str(img_id)]
        return torch.from_numpy(img_array)
    
    def _read_img_tensor_random_mask(self, img_id):
        img_array = self.imgs[str(img_id)]
        x_start = random.randrange(224)
        y_start = random.randrange(224)

        for x in range(x_start, min(224, x_start + 32)):
            for y in range(y_start, min(224, y_start + 32)):
                r,g,b = img_array[0][x][y], img_array[1][x][y], img_array[2][x][y]
                img_array[0][x][y] = 0
                img_array[1][x][y] = 0
                img_array[2][x][y] = 0
        
        return torch.from_numpy(img_array)
    
    def _read_vg_img_tensor_from_npzfile(self, img_id):
        img_array = self.vg_imgs[str(img_id)]
        return torch.from_numpy(img_array)
    
    def _read_mask_img_tensor_from_npzfile(self, img_id):
        img_array = self.vg_mask_imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, image, caption, segs, seg_preds = self.samples[idx]

        img_id = image.split('/')[-1].split('.jpg')[0]
        images = self._read_img_tensor_from_npzfile(img_idx)
        texts = tokenize([str(caption)])[0]

        image_objects = self.img_obj_dict[img_id]
        random.shuffle(image_objects)

        da_images = []
        da_images_masks = []
        da_texts = []
        da_cnt = 0

        for index, obj in enumerate(image_objects):
            obj_img_items = self.img_mask_dict[obj]
            random.shuffle(obj_img_items)
            obj_img_ids = [obj_id["image_id"] for obj_id in obj_img_items]

            if len(obj_img_ids) == 0:
                continue
            
            if da_cnt >= 1:
                break
            
            obj_text = self.prompt.format(object=obj)
            obj_text = tokenize([str(obj_text)])[0]

            obj_mask_cnt = 0
            for index, img_item in enumerate(obj_img_items):
                if obj_mask_cnt >= 1:
                    break                
                vg_img_id = img_item["image_id"]
                img_obj_region = img_item["object"]

                if len(img_item["object"]) == 0:
                    continue

                obj_img_tensor = self._read_vg_img_tensor_from_npzfile(str(vg_img_id))

                mask_img_id = "{}_{}".format(obj, vg_img_id)
                obj_mask_img_tensor = self._read_mask_img_tensor_from_npzfile(str(mask_img_id))

                da_images.append(obj_img_tensor.unsqueeze(0))
                da_images_masks.append(obj_mask_img_tensor.unsqueeze(0))
                da_texts.append(obj_text.unsqueeze(0))

                da_cnt += 1
                obj_mask_cnt += 1
                
        if len(da_images) == 0:
            da_images.append(image.unsqueeze(0))
            da_texts.append(text.unsqueeze(0))

            image_random_mask = self._read_img_tensor_random_mask(str(img_idx))
            da_images_masks.append(image_random_mask.unsqueeze(0))

        da_images = torch.cat(da_images)
        da_texts = torch.cat(da_texts)
        da_images_masks = torch.cat(da_images_masks)

        da_vg_images_masks = []
        da_vg_cnt = 0

        for index, obj in enumerate(seg_preds):
            if da_vg_cnt >= 1:
                break
            
            da_vg_id = "{}_{}".format(img_id, obj[0][0])
            obj_img_tensor = self._read_vg_mask_img_tensor_from_npzfile(str(da_vg_id))
            da_vg_images_masks.append(obj_img_tensor)

            da_vg_cnt += 1

        if len(da_vg_images_masks) == 0:
            image_random_mask = self._read_img_tensor_random_mask(str(img_idx))
            da_vg_images_masks.append(image_random_mask)

        txt_objs_prompts = [self.prompt.format(object=txt_obj) for txt_obj in segs if len(txt_obj.split(' ')) <= 7]
        txt_objs_masks = [caption.replace(txt_obj, '[MASK]') for txt_obj in segs if len(txt_obj.split(' ')) <= 7]
        prompt_len = len(txt_objs_prompts)

        if prompt_len < 5:
            for obj_idx in range(prompt_len, 5):
                txt_objs_prompts.append("")
                txt_objs_masks.append("")
        elif prompt_len > 5:
            txt_objs_prompts = txt_objs_prompts[:5]
            txt_objs_masks = txt_objs_masks[:5]
                    
        text_prompts = tokenize(txt_objs_prompts)
        text_masks = tokenize(txt_objs_masks)

        if self.is_object:
            image_objects = self.img_obj_dict[img_id]
            if len(image_objects) < 32:
                for obj_idx in range(len(image_objects), 32):
                    image_objects.append("-1")

            return images, image_objects, da_images, da_texts, da_images_masks, da_vg_images_masks[0], texts, text_prompts, text_masks
        else:
            return images, da_images, da_texts, da_images_masks, da_vg_images_masks[0], texts, text_prompts, text_masks

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def get_json_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    split_id = "train" if is_train else "val"
    assert input_filename

    if is_train and args.is_prompt and args.is_mask and args.is_da_loss and args.is_da_mask and args.is_vg:
        dataset = Json_All_Dataset(
        input_filename,
        preprocess_fn,
        img_key=args.img_key,
        caption_key=args.caption_key,
        split=split_id)
    else:
        dataset = JsonDataset(
            input_filename,
            preprocess_fn,
            img_key=args.img_key,
            caption_key=args.caption_key,
            split=split_id)

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
    if dataset_type == "json":
        return get_json_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['json']:
            return get_json_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    print(args.dataset_type)

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True)
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False)

    return data
