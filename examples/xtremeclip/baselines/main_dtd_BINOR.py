import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils_eurosat import *
IMAGE_ROOT = '/home/moming/data/cache/dtd'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/moming/code/VITCLIP/configs/dtd.yaml', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def clip_classifier(classnames, template, clip_model):
    device_id = clip_model.positional_embedding.device
    clip_weights = []

    for classname in classnames:
        # Tokenize the prompts
        classname = classname.replace('_', ' ')
        texts = [t.format(classname) for t in template]
        texts = clip.tokenize(texts).cuda(device_id)
        # prompt ensemble for ImageNet
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding/class_embedding.norm()
        clip_weights.append(class_embedding)

    clip_weights = torch.stack(clip_weights, dim=1).cuda(device_id)
    return clip_weights

def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']

    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, test_loader, test_labels, dataset,
                      clip_model, train_loader_F):
    device_id = clip_model.positional_embedding.device
    # Enable the cached keys to be learnable

    include = lambda n: "bn" in n or "ln" in n or "bias" in n
    named_parameters = list(clip_model.named_parameters())

    tune_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    fix_params = [p for n, p in named_parameters if not include(n)]
    optimizer = torch.optim.AdamW(
        [
         {'params': tune_params, 'lr': 5e-7},
         {'params': fix_params, 'lr': 0},
         ],
        lr=2e-3, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(30):
        # Train
        clip_model.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print('Train Epoch: {:} / {:}'.format(train_idx, 30))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(device_id), target.cuda(device_id)
            # with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
            tip_logits = temp.exp() * image_features @ clip_weights

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))

        # Eval
        clip_model.eval()
        test_features = []
        dump_img_idx = 0
        num_text = len(test_loader)
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images, target = images.cuda(device_id), target.cuda(device_id)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            test_features.append(image_features.cpu())
            if len(test_features) % 8 == 0 and len(test_features) != 0 or dump_img_idx * 8 * 64 + 64 > num_text:
                torch.save(torch.cat(test_features, dim=0), IMAGE_ROOT + "test_img_" + str(dump_img_idx) + ".pt")
                test_features = []
                dump_img_idx += 1

        base_dir = IMAGE_ROOT + "test_img_" + "%s" + ".pt"
        image_embeds = []
        for n in range(dump_img_idx):
            image_embeds.append(torch.load(base_dir % str(n)))
        image_embeds = torch.cat(image_embeds, dim=0).to(device_id)

        clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)
        tip_logits = image_embeds @ clip_weights
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            # torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    # adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")



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
    clip_model, preprocess = clip.load(cfg['backbone'], 'cuda:7')
    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                   shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                    shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform,
                                           is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=8, tfm=train_tranform, is_train=True,
                                       shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values, clss_head = build_cache_model(cfg, clip_model, train_loader_cache, clip_weights)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # # ------------------------------------------ Tip-Adapter ------------------------------------------
    # run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

    # # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, test_loader, test_labels, dataset,
                      clip_model, train_loader_F)


if __name__ == '__main__':
    main()