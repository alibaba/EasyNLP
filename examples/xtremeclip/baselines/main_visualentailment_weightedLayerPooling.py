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
from utils_visualentailment import *
import base64
from torchvision import transforms
from PIL import Image, ImageFile

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([0,0,0,0,0,0,0.1,0.9], dtype=torch.float)
            )

    def forward(self, features):
        if type(features) == dict:
            features = [v[:, 0, :] for k, v in features.items()]
            all_layer_embedding = torch.stack(features)
        else:
            all_layer_embedding = features
        # all_layer_embedding = torch.stack(features)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size()).to(all_layer_embedding.device)
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/moming/code/Tip-Adapter/configs/visualentailment.yaml', dest='config',
                        help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels):
    
    # Zero-shot CLIP
    beta = cfg['init_beta']
    affinity = test_features @ cache_keys.t()
    acc = cls_acc(affinity, test_labels)
    # cache_logits = affinity @ cache_values
    # acc = cls_acc(cache_logits, test_labels)
    # for adapter_layer, layer_emb in zip(cache_layers, test_layers):
    #     layer_logits = layer_emb @ adapter_layer.t()
    #     layer_acc = cls_acc(layer_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    # _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels)


def run_tip_adapter_F(cfg, test_img_layers, test_txt_layers, test_features, test_labels, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    device_id = clip_model.positional_embedding.device
    adapter = nn.Linear(512 * 5, 3, bias=False).to(torch.float32)
    # adapter.weight = nn.Parameter(cache_keys.t())
    # adapter.weight = nn.Parameter(cache_keys)
    adapter.to(device_id)
    # layer_adapter = nn.Linear(cache_layers.shape[0], cache_layers.shape[1], bias=False).to(clip_model.dtype).cuda()
    # layer_adapter.weight = nn.Parameter(cache_layers)
    
    # optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    weighted_sum = WeightedLayerPooling(12)

    optimizer = torch.optim.AdamW(
        [
            {"params": clip_model.parameters(), 'lr': 5e-7},
            {"params": adapter.parameters(), "lr": 2e-3},
            {"params": weighted_sum.parameters(), "lr": 2e-7}
        ],
        lr=cfg['lr'],
        betas=(0.9, 0.999),
        eps=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        clip_model.train()
        adapter.train()
        weighted_sum.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, hypothesis, premise, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.to(device_id), target.to(device_id)
            # with torch.no_grad():
            images = images.to(device_id)
            image_features, layer_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            for k, v in layer_features.items():
                v = v / v.norm(dim=-1, keepdim=True)
                layer_features[k] = v.permute(1, 0, -1)
            weighted_img_emb = weighted_sum(layer_features)
            hypo_tensor = clip.tokenize(hypothesis).to(device_id)
            hypo_feature, txt_layer_cache = clip_model.encode_text(hypo_tensor, output_all_layers=True)
            hypo_feature = hypo_feature / hypo_feature.norm(dim=-1, keepdim=True)
            for k, v in txt_layer_cache.items():
                v = v / v.norm(dim=-1, keepdim=True)
                txt_layer_cache[k] = v.permute(1, 0, -1)
            weighted_txt_emb = weighted_sum(txt_layer_cache)
            adapter_weights = fusion_feat(weighted_txt_emb, weighted_img_emb)
            # adapter_weights = torch.cat([weighted_txt_emb, weighted_img_emb], dim=-1)

            # affinity = adapter(image_features)
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            # affinity = adapter(adapter_weights)
            # cache_logits = affinity @ cache_values
            cache_logits = adapter(adapter_weights)

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
        adapter.eval()
        clip_model.eval()
        weighted_sum.eval()
        # affinity = adapter(test_features)
        # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # affinity = adapter(test_features)
        # cache_logits = affinity @ cache_values
        # weighted_sum_test_txt = weighted_sum(test_txt_layers)
        # weighted_sum_test_img = weighted_sum(test_img_layers)
        # weighted_sum_test = torch.cat([weighted_sum_test_txt, weighted_sum_test_img], dim=-1)
        weighted_test_txt = weighted_sum(test_txt_layers)
        weighted_test_img = weighted_sum(test_img_layers)
        weighted_sum_test = fusion_feat(weighted_test_txt, weighted_test_img)
        cache_logits = adapter(weighted_sum_test)

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
    clip_model, preprocess = clip.load(cfg['backbone'], "cuda:0")
    # clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing SNLI_VE dataset.")
    train_datasets, dev_datasets = load_data_file(cfg['root_path'])

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
    train_loader_F = torch.utils.data.DataLoader(train_datasets, batch_size=256, num_workers=8, shuffle=True,
                                                 collate_fn=process_image_from_ofa)

    # Construct the cache model by few-shot training set
    # print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels, test_img_layers, test_txt_layers = pre_load_features(cfg, "test", clip_model, test_loader, load_cache_layer=True)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, test_img_layers, test_txt_layers, test_features, test_labels, clip_model, train_loader_F)
           

if __name__ == '__main__':
    main()