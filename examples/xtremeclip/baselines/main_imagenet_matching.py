import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils_imagenet_matching import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/moming/code/VITCLIP/configs/imagenet.yaml', dest='config',
                        help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    # for adapter_layer, layer_emb in zip(cache_layers, test_layers):
    #     layer_logits = layer_emb @ adapter_layer.t()
    #     layer_acc = cls_acc(layer_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    # _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)


def run_tip_adapter_F(cfg, clss_head, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F, max_epoch=200):
    
    # Enable the cached keys to be learnable
    device_id = clip_model.positional_embedding.device
    adapter = nn.Linear(clss_head.shape[1], clss_head.shape[0], bias=False).to(clip_model.dtype).cuda(device_id)
    adapter.weight = nn.Parameter(clss_head)
    device_id = clip_model.positional_embedding.device

    optimizer = torch.optim.AdamW(
        [{'params': adapter.parameters()},
         # {'params': clip_weights},
         ],
        lr=5e-3, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(device_id), target.cuda(device_id)
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            img_feat = image_features.unsqueeze(1).repeat(1, clip_weights.size(-1), 1)
            txt_feat = clip_weights.permute(1, 0).unsqueeze(0).repeat(images.size(0), 1, 1)
            final_feat = fusion_feat(txt_feat, img_feat)#torch.cat([txt_feat, img_feat], dim=-1)
            logits = adapter(final_feat).squeeze(-1)
            logits = torch.stack([torch.diag(l) for l in logits], dim=0)

            log_prob = torch.softmax(logits, dim=-1)  # torch.softmax(logits, dim=-1)  #logits # torch.log_softmax(logits, dim=-1)
            pos_score = log_prob.gather(-1, target.unsqueeze(-1).repeat(1, logits.size(1)))
            ones = torch.ones(pos_score.size()).cuda(pos_score.device)
            loss_func = torch.nn.MarginRankingLoss(0.0)
            contr_loss = loss_func(pos_score, log_prob, ones)
            loss = F.cross_entropy(logits, target) + contr_loss

            acc = cls_acc(logits, target)
            correct_samples += acc / 100 * len(logits)
            all_samples += len(logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        # test_img_feat = test_features.unsqueeze(1).repeat(1, clip_weights.size(-1), 1)
        # test_txt_feat = clip_weights.permute(1, 0).unsqueeze(0).repeat(test_features.size(0), 1, 1)
        bs = 50
        test_txt_feat = clip_weights.permute(1, 0).unsqueeze(0).repeat(bs, 1, 1)
        accs = []
        for t in range(0,  test_features.size(0), bs):
            test_img_feat = test_features[t: t+bs].unsqueeze(1).repeat(1, clip_weights.size(-1), 1)
            test_in = fusion_feat(test_txt_feat, test_img_feat)#torch.cat([test_txt_feat, test_img_feat], dim=-1)
            test_b_logits = adapter(test_in).squeeze(-1)
            test_logits = torch.stack([torch.diag(l) for l in test_b_logits], dim=0)
            b_acc = cls_acc(test_logits, test_labels[t: t+bs])
            accs.append(b_acc)

        acc = 100 * sum(accs) / test_features.size(0)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    # _ = search_hp(cfg, None, None, test_features, test_labels, clip_weights, adapter=adapter)


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
    clip_model, preprocess = clip.load(cfg['backbone'], 'cuda:1')
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=16, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=16, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values, clss_head = build_cache_model(cfg, clip_model, train_loader_cache, clip_weights)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, clss_head, test_features, test_labels, clip_weights,
                      clip_model, train_loader_F)
           

if __name__ == '__main__':
    main()