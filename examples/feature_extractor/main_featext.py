from re import S
import sys
import os

sys.path.append('./')
sys.path.append('./easynlp/appzoo/')
sys.path.append('./easynlp/appzoo/sequence_classification/')

print('*'*50)
print('running local main...\n')

import braceexpand
from easynlp.utils import initialize_easynlp, get_args
from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from transformers import CLIPModel, CLIPProcessor
import webdataset as wds


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


def process_folder(rank, url_folder, url_index, output):
    url = "/data/oss_bucket_0/data/wukong/%s/0000%d.tar" % (url_folder, url_index)
    print('\n\n')
    print('Is dir exists:', os.path.isdir(url), url)
    print('Is File exists:', os.path.isfile(url), url)
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
    os.system(f'dir {url_dataset}')
    file_len = 0
    
    for batch_index, batch in enumerate(url_dataset):
        file_len += 1
        if batch_index == 0:
            path = url_folder + '/0000' + str(url_index) + '.tar'
            print(rank, url_dataset)
            print(rank, path)

            # imgs, infos = batch
            # batch_objects = []
            # for img, info in zip(imgs, infos):
            #     sample_info = (info['key'], info['width'], info['height'], path, info['caption'], img)
            #     batch_objects.append(sample_info)

            #     batch_features = extract_image_features(batch_objects)
            #     with common_io.table.TableWriter(output, slice_id=rank) as f:
            #         f.write(batch_features, (0,1,2,3,4,5,6))
        
        if batch_index % 50 == 0:
            print("{} batch objects extracts".format(str(file_len)))

    print("{} batch objects extracts".format(file_len))
            
    print("Finished writing {} batch features for url {}".format(str(file_len), '0000{}.tar'.format(str(url_index))))
    print("Finished writing {} batch features for url {}, worker {}".format(str(file_len), url, str(rank)))
    
    return file_len


def feature_extractor(rank, world_size, output):
    start_index = rank * 2
    end_index = start_index + 2

    print(f"Running DDP on rank {rank}, world {world_size}, start {start_index} end {end_index}.")
    print("CLIP ViT-L-14 model loaded from transformers!")

    # model.to(device)

    batch_size = 1024
    print("Start processing with batch size {}".format(str(batch_size)))

    for process_index in range(start_index, end_index):
        url_folder = "output_{}".format(str(process_index))
        print("Processing current url folder file {}...".format(url_folder))
        item_cnt = 0
        for file_index in range(5):
            file_item_length = process_folder(rank, url_folder, file_index, output)
            item_cnt += file_item_length
                        
        print("Finished processing features for folder {} with {} items in total.".format(url_folder, str(item_cnt)))


def feature_extractor2(rank):
    file_len = 0
    
    urls = list(braceexpand.braceexpand("/data/oss_bucket_0/data/wukong/output_{0..4}/{00000..00003}.tar"))
    dataset = wds.WebDataset(urls)

    dataloader = torch.utils.data.DataLoader(
            dataset, 
            num_workers=1, 
            batch_size=128)

    for batch_index, batch in enumerate(dataloader):
        if file_len == 0:
            print(rank, dataset)
            print(rank, urls)

        file_len += 1

        if file_len % 50 == 0:
            print(f"Processed {file_len} batches, {rank}")
    
    print(f"Processed {file_len} batches, {rank}")

if __name__ == "__main__":
    print('log: starts to init...\n')
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_DEBUG"] = "INFO"

    initialize_easynlp()
    args = get_args()

    device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"

    print(args)
    print(args.rank)
    print(args.local_rank)
    print(args.world_size)
    
    # original
    # feature_extractor(args.rank, args.world_size, output)

    # split by nodes
    feature_extractor2(args.rank)
