from __future__ import print_function

import io
import oss2
import torch
import threading

import os
import pandas as pd

from io import StringIO
from PIL import Image

from transformers import CLIPModel, CLIPProcessor

import torch.distributed as dist
import torch.multiprocessing as mp

import webdataset as wds

import common_io
import argparse

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop


def extract_image_features(all_image_objects):
    image_inputs = processor(images=[object_item[5] for object_item in all_image_objects], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs).to(device)
            
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()

    feature_dict = []
    for image_feature, image_object in zip(image_features, all_image_objects):
        image_feature_str = str(image_feature.tolist())[1:-1]
        image_object =  image_object[:5] + (image_feature_str,)
        feature_dict.append(image_object)

    return feature_dict


def _convert_to_rgb(image):
    return image.convert('RGB')


def process_folder(url_folder, url_index, output):
    url = oss_prefix.format(folder=url_folder, item=str(url_index))
    print("Processing file {} with url {}".format(url_folder, url))

    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

    url_dataset = (wds.WebDataset(url).decode("pil").to_tuple("jpg", "json").map_tuple(transform))
    url_dataloader = torch.utils.data.DataLoader(url_dataset.batched(1024), num_workers=8, batch_size=None)
    path = url_folder + '/0000' + str(url_index) + '.tar'
    file_features = []
    for batch_index, batch in enumerate(url_dataloader):
        imgs, infos = batch
        batch_objects = []
        for img, info in zip(imgs, infos):
            sample_info = (info['key'], info['width'], info['height'], path, info['caption'], img)
            batch_objects.append(sample_info)
        
        print("{} batch objects extracts".format(str(len(batch_objects))))

        batch_features = extract_image_features(batch_objects)
        id_slice = rank

        with common_io.table.TableWriter(output, slice_id=id_slice) as f:
            f.write(batch_features, (0,1,2,3,4,5))
                
        print("Image features successfully written in odps for batch {}.".format(str(batch_index))) 
        file_features.append(batch_features)
            
    print("Finished writing {} batch features for url {}".format(str(len(file_features)), '0000{}.tar'.format(str(url_index))))

    return len(file_features)


def feature_extractor(rank, world_size, output):
    print(f"Running basic DDP example on rank {rank}.")

    start_index = rank * 8
    end_index = start_index + 8

    print("CLIP ViT-L-14 model loaded from transformers!")

    model.to(device)

    batch_size = 1024
    print("Start processing with batch size {}".format(str(batch_size)))

    for process_index in range(start_index, end_index):
        url_folder = "output_{}".format(str(process_index))
        print("Processing current url folder file {}...".format(url_folder))
        item_cnt = 0
        for file_index in range(5):
            file_item_length = process_folder(url_folder, file_index, output)
            item_cnt += file_item_length
                        
        print("Finished processing features for folder {} with {} items in total.".format(url_folder, str(item_cnt)))
        

if __name__ == '__main__':
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print("Wordsize: ", world_size)
    print("Rank: ", rank)
    
    oss_prefix = "https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp/datasets/wukong/{folder}/0000{item}.tar"

    parser = argparse.ArgumentParser(description='PyTorch table IO')
    parser.add_argument('--tables', default="", type=str, help='ODPS input table names')
    parser.add_argument('--outputs', default="", type=str, help='ODPS output table names')
    
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    feature_extractor(rank, world_size, args.outputs)