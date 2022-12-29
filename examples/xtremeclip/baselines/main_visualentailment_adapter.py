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
from utils_visualentailment_adapter import *
import base64
from torchvision import transforms
from PIL import Image, ImageFile

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/moming/code/VITCLIP/configs/visualentailment.yaml', dest='config',
                        help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_txt_feature, test_labels, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    device_id = clip_model.positional_embedding.device
    adapter_up = nn.Linear(512, 128, bias=False).to(clip_model.dtype)
    adapter_down = nn.Linear(128, 512, bias=False).to(clip_model.dtype)
    adapter_txt_up = nn.Linear(512, 128, bias=False).to(clip_model.dtype)
    adapter_txt_down = nn.Linear(128, 512, bias=False).to(clip_model.dtype)
    clss_head = nn.Linear(512*5, 3, bias=False).to(clip_model.dtype)
    # adapter.weight = nn.Parameter(cache_keys.t())
    # adapter.weight = nn.Parameter(cache_keys)
    adapter_up.to(device_id)
    adapter_down.to(device_id)
    adapter_txt_up.to(device_id)
    adapter_txt_down.to(device_id)
    clss_head.to(device_id)
    # layer_adapter = nn.Linear(cache_layers.shape[0], cache_layers.shape[1], bias=False).to(clip_model.dtype).cuda()
    # layer_adapter.weight = nn.Parameter(cache_layers)
    
    # optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    include = lambda n: "bias" in n
    named_parameters = list(clip_model.named_parameters())

    tune_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    fix_params = [p for n, p in named_parameters if not include(n)]
    optimizer = torch.optim.AdamW(
        [
            {"params": tune_params, 'lr': 0.0},
            {"params": fix_params, "lr": 0.0},
            {"params": adapter_up.parameters(), "lr": 2e-3},
            {"params": adapter_down.parameters(), "lr": 2e-3},
            {"params": adapter_txt_up.parameters(), "lr": 2e-3},
            {"params": adapter_txt_down.parameters(), "lr": 2e-3},
            {"params": clss_head.parameters(), "lr": 2e-3}
        ],
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        eps=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    relu = torch.nn.ReLU()
    best_acc, best_epoch = 0.0, 0
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    for train_idx in range(cfg['train_epoch']):
        # Train
        clip_model.train()
        adapter_down.train()
        adapter_up.train()
        adapter_txt_up.train()
        adapter_txt_down.train()
        clss_head.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, hypothesis, premise, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(device_id), target.to(device_id)
            # with torch.no_grad():
            image_features = encode_img(clip_model, images, device_id)
            adapt_image_features = adapter_up(image_features)
            adapt_image_features = relu(adapt_image_features)
            adapt_image_features = adapter_down(adapt_image_features)
            adapt_image_features = relu(adapt_image_features)
            image_features = 0.2 * adapt_image_features + 0.8 * image_features
            hypo_feature = encode_txt(clip_model, hypothesis, device_id)
            # adapt_hypo_feature = adapter_txt_up(hypo_feature)
            # adapt_hypo_feature = relu(adapt_hypo_feature)
            # adapt_hypo_feature = adapter_txt_down(adapt_hypo_feature)
            # adapt_hypo_feature = relu(adapt_hypo_feature)
            # hypo_feature = 0.2 * adapt_hypo_feature + 0.8 * hypo_feature
            adapter_weights = fusion_feat(hypo_feature, image_features)

            # affinity = adapter(image_features)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # affinity = adapter(adapter_weights)
            # cache_logits = affinity @ cache_values

            cache_logits = clss_head(adapter_weights)
            loss = F.cross_entropy(cache_logits, target)

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
        adapter_up.eval()
        adapter_down.eval()
        adapter_txt_down.eval()
        adapter_txt_up.eval()
        clip_model.eval()
        clss_head.eval()
        adapt_img_feat = adapter_up(test_features)
        adapt_img_feat = relu(adapt_img_feat)
        adapt_img_feat = adapter_down(adapt_img_feat)
        adapt_img_feat = relu(adapt_img_feat)
        adapt_img_feat = 0.2 * adapt_img_feat + 0.8 * test_features
        adapt_txt_feat = test_txt_feature#adapter_txt_up(test_txt_feature)
        # adapt_txt_feat = relu(adapt_txt_feat)
        # adapt_txt_feat = adapter_txt_down(adapt_txt_feat)
        # adapt_txt_feat = 0.2 * adapt_txt_feat + 0.8 * test_txt_feature
        adapter_weights = fusion_feat(adapt_txt_feat, adapt_img_feat)
        cache_logits = clss_head(adapter_weights)
        # affinity = adapter(test_features)
        # cache_logits = affinity @ cache_values
        # cache_logits = adapter(test_features)
        acc = cls_acc(cache_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter_up.weight, cfg['cache_dir'] + "/best_img_up_" + str(cfg['shots']) + "shots.pt")
            torch.save(adapter_down.weight, cfg['cache_dir'] + "/best_img_down_" + str(cfg['shots']) + "shots.pt")
            torch.save(adapter_txt_up.weight, cfg['cache_dir'] + "/best_txt_up_" + str(cfg['shots']) + "shots.pt")
            torch.save(adapter_txt_down.weight, cfg['cache_dir'] + "/best_txt_down_" + str(cfg['shots']) + "shots.pt")
    
    adapter_up.weight = torch.load(cfg['cache_dir'] + "/best_img_up_" + str(cfg['shots']) + "shots.pt")
    adapter_down.weight = torch.load(cfg['cache_dir'] + "/best_img_down_" + str(cfg['shots']) + "shots.pt")
    adapter_txt_up.weight = torch.load(cfg['cache_dir'] + "/best_txt_up_" + str(cfg['shots']) + "shots.pt")
    adapter_txt_down.weight = torch.load(cfg['cache_dir'] + "/best_txt_down_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    # _ = search_hp(cfg, None, cache_values, test_features, test_labels, adapter=adapter, clss_head=clss_head)


def load_data_file(data_file):
    train_path = data_file % "train"
    dev_path = data_file % "dev"
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
    clip_model, preprocess = clip.load(cfg['backbone'], "cuda:2")
    # clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing SNLI_VE dataset.")
    train_datasets, dev_datasets = load_data_file(cfg['root_path'])
    train_datasets = train_datasets[:2000]
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
        hypothesis = []
        premise = []
        labels = []
        lines = [s.split('\t') for s in samples]
        for l in lines:
            img_str = l[2]
            image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str)))
            image = patch_resize_transform(image)
            images.append(image.unsqueeze(0))
            hypothesis.append(l[3])
            premise.append(l[4])
            if 'contradiction' in l[5]:
                labels.append(0)
            elif "neutral" in l[5]:
                labels.append(1)
            else:
                labels.append(2)
        return torch.cat(images, dim=0), hypothesis, premise, torch.tensor(labels).to(torch.int64)



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
    test_features, test_txt_feature, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_txt_feature, test_labels, clip_model, train_loader_F)
           

if __name__ == '__main__':
    main()