import os
import random
import argparse
import yaml
from tqdm import tqdm
from io import BytesIO
import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils_vqa import *
import base64
from torchvision import transforms
from PIL import Image, ImageFile

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/moming/code/VITCLIP/configs/vqa.yaml', dest='config',
                        help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels):
    
    # Zero-shot CLIP
    beta = cfg['init_beta']
    # affinity = test_features @ cache_keys.t()
    affinity = test_features @ cache_keys.t()
    # affinity = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    acc = cls_acc(affinity, test_labels)
    # cache_logits = affinity @ cache_values
    # acc = cls_acc(cache_logits, test_labels)
    # for adapter_layer, layer_emb in zip(cache_layers, test_layers):
    #     layer_logits = layer_emb @ adapter_layer.t()
    #     layer_acc = cls_acc(layer_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    # _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels)


def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    device_id = clip_model.positional_embedding.device
    adapter = nn.Linear(cache_keys.shape[1], cache_keys.shape[0], bias=False).to(clip_model.dtype).to(device_id)
    adapter.weight = nn.Parameter(cache_keys)
    # adapter.weight = nn.Parameter(cache_keys)
    # layer_adapter = nn.Linear(cache_layers.shape[0], cache_layers.shape[1], bias=False).to(clip_model.dtype).cuda()
    # layer_adapter.weight = nn.Parameter(cache_layers)

    # optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-3, eps=1e-4)
    # include = lambda n: "bias" in n
    # named_parameters = list(clip_model.named_parameters())
    # for n, p in named_parameters:
    #     if not include(n):
    #         p.requires_grad = False
    # tune_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            # {"params": tune_params},
            # {"params": rest_params, "weight_decay": 0.001},
            {"params": adapter.parameters(), 'lr': cfg['lr']} #2e-3 59.02 1e-3 59.07
        ],
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        eps=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        clip_model.eval()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, questions, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(device_id), target.to(device_id)
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                hypo_tensor = clip.tokenize(questions).to(device_id)
                hypo_feature = clip_model.encode_text(hypo_tensor)
                hypo_feature /= hypo_feature.norm(dim=-1, keepdim=True)
                plus_feature = hypo_feature + image_features
                minus_feature = image_features - hypo_feature
                times_feature = image_features * hypo_feature
                adapter_weights = torch.cat([image_features, hypo_feature, plus_feature, minus_feature, times_feature], dim=-1)
                # adapter_weights = torch.cat([image_features, hypo_feature], dim=-1)


            # affinity = adapter(adapter_weights)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # affinity = adapter(adapter_weights)
            # cache_logits = affinity @ cache_values
            cache_logits = adapter(adapter_weights)

            # cache_logits = torch.softmax(cache_logits, dim=-1)

            pos_score = cache_logits.gather(-1, target.unsqueeze(-1))
            ones = torch.ones(pos_score.size()).cuda(cache_logits.device)
            loss_func = torch.nn.MarginRankingLoss(0.0, reduction='sum') # , reduction='sum'
            contr_loss = loss_func(pos_score.repeat(1, 2), cache_logits, ones)
            loss = F.cross_entropy(cache_logits, target) + contr_loss

            acc = cls_acc(cache_logits, target)
            correct_samples += acc / 100 * len(cache_logits)
            all_samples += len(cache_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()
        clip_model.eval()
        # affinity = adapter(test_features)
        # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # affinity = adapter(test_features)
        # cache_logits = affinity @ cache_values
        cache_logits = adapter(test_features)
        acc = cls_acc(cache_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    # _ = search_hp(cfg, None, cache_values, test_features, test_labels, adapter=adapter)

def load_data_file(data_file):
    train_path = data_file % "train"
    dev_path = data_file % "val"
    train_lines = open(train_path, 'r').readlines()
    dev_lines = open(dev_path, 'r').readlines()
    return train_lines, dev_lines

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'], cfg['gpu'])
    # clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing VQA dataset.")
    train_datasets, dev_datasets = load_data_file(cfg['root_path'])
    train_datasets = train_datasets[:10000]
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    patch_image_size = 224
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.Resampling.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def process_image_from_ofa(samples):
        images = []
        questions = []
        labels = []
        lines = [s.split('\t') for s in samples]
        for l in lines:
            img_str = l[4]
            image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str)))
            image = patch_resize_transform(image)
            images.append(image.unsqueeze(0))
            questions.append(l[2])
            if 'no' in l[3]:
                labels.append(0)
            elif "yes" in l[3]:
                labels.append(1)
        return torch.cat(images, dim=0), questions, torch.tensor(labels).to(torch.int64)



    test_loader = torch.utils.data.DataLoader(dev_datasets, batch_size=64, num_workers=8, shuffle=False,
                                              collate_fn=process_image_from_ofa)
    train_loader_cache = torch.utils.data.DataLoader(train_datasets, batch_size=256, num_workers=8, shuffle=False,
                                                     collate_fn=process_image_from_ofa)
    train_loader_F = torch.utils.data.DataLoader(train_datasets, batch_size=16, num_workers=8, shuffle=True,
                                                 collate_fn=process_image_from_ofa)

    # Construct the cache model by few-shot training set
    # print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_model, train_loader_F)
           

if __name__ == '__main__':
    main()