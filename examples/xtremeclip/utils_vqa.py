from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from utils_visualentailment import fusion_feat
import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    device_id = clip_model.positional_embedding.device

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_textual = []
        cache_values = []
        layers = {}
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                train_textual_features = []
                cache_layers = {}
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, questions, targets) in enumerate(tqdm(train_loader_cache)):
                    images, target = images.to(device_id), targets.to(device_id)
                    image_features = clip_model.encode_image(images)
                    hypo_tensor = clip.tokenize(questions).to(device_id)
                    hypo_feature = clip_model.encode_text(hypo_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    train_features.append(image_features)
                    hypo_feature /= hypo_feature.norm(dim=-1, keepdim=True)
                    train_textual_features.append(hypo_feature)
                    # for idx, emb in image_caches.items():
                    #     emb /= emb.norm(dim=-1, keepdim=True)
                    #     emb = emb.permute(1, 0, -1).cpu()
                    #     cls_emb = emb[:, 0, :]
                    #     layers[idx].append(cls_emb)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                cache_textual.append(torch.cat(train_textual_features, dim=0).unsqueeze(0))
                for idx, emb in cache_layers.items():
                    layers[idx] = emb.unsqueeze(0) if idx not in layers.keys() else torch.cat([layers[idx], emb.unsqueeze(0)])

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_textual = torch.cat(cache_textual, dim=0).mean(dim=0)
        cache_textual /= cache_textual.norm(dim=-1, keepdim=True)
        cache_textual = cache_textual.permute(1, 0)

        plus_feature = cache_textual + cache_keys
        minus_feature = cache_keys - cache_textual
        times_feature = cache_keys * cache_textual

        # adapter_weights = torch.cat([cache_keys, cache_textual], dim=0)
        adapter_weights = torch.cat([cache_keys, cache_textual, plus_feature, minus_feature, times_feature], dim=0)
        cls_weights = [[] for _ in range(2)]
        for lbl, emb in zip(torch.cat(cache_values, dim=0).tolist(), adapter_weights.permute(1, 0)):
            cls_weights[lbl].append(emb)
        adapter_weights = torch.stack([torch.stack(c, dim=0).mean(dim=0) for c in cls_weights])
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        # for i, emb in layers.items():
        #     layers[i] = emb.mean(dim = 0)
        #     torch.save(layers[i].cpu(), cfg['cache_dir'] + '/layer_' + str(i) + "_cache.pt")
        torch.save(adapter_weights.cpu(), cfg['cache_dir'] + '/keys_' + 'vqa' + ".pt")
        torch.save(cache_values.cpu(), cfg['cache_dir'] + '/values_' + 'vqa' + ".pt")

    else:
        # layer_num = len(clip_model.visual.transformer.resblocks)
        # layers = []
        # for i in range(layer_num):
        #     layer = torch.load(cfg['cache_dir'] + '/layer_' + str(i) + "_cache.pt")
        #     layers.append(layer.to(device_id))
        # layers = torch.load(cfg['cache_dir'] + '/layer_' + str(layer_num - 1) + "_cache.pt").to(device_id)
        adapter_weights = torch.load(cfg['cache_dir'] + '/keys_' + 'vqa' + ".pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + 'vqa' + ".pt")

    return adapter_weights.to(device_id), cache_values.to(device_id)
    # return cache_keys.to(device_id), cache_values.to(device_id), layers.to(device_id)

def encode_img(clip_model, images, device_id):
    images = images.to(device_id)
    image_features = clip_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

def encode_txt(clip, clip_model, questions, device_id):
    hypo_tensor = clip.tokenize(questions).to(device_id)
    hypo_feature = clip_model.encode_text(hypo_tensor)
    hypo_feature = hypo_feature / hypo_feature.norm(dim=-1, keepdim=True)
    return hypo_feature

def fusion_feat(hypo_feature, image_features):
    plus_feature = hypo_feature + image_features
    minus_feature = image_features - hypo_feature
    times_feature = image_features * hypo_feature
    adapter_weights = torch.cat([image_features, hypo_feature, plus_feature, minus_feature, times_feature], dim=-1)
    return adapter_weights

def pre_load_features(cfg, split, clip_model, loader):

    layer_num = len(clip_model.visual.transformer.resblocks)
    device_id = clip_model.positional_embedding.device

    if cfg['load_pre_feat'] == False:
        features, textual_feature, labels, layers = [], [], [], []
        layers = [[] for _ in range(layer_num)]
        with torch.no_grad():
            for i, (images, questions, targets) in enumerate(tqdm(loader)):
                images, target = images.to(device_id), targets.to(device_id)
                image_features = clip_model.encode_image(images)
                hypo_tensor = clip.tokenize(questions).to(device_id)
                hypo_feature = clip_model.encode_text(hypo_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                hypo_feature /= hypo_feature.norm(dim=-1, keepdim=True)
                # for idx, emb in image_caches.items():
                #     emb /= emb.norm(dim=-1, keepdim=True)
                #     emb = emb.permute(1, 0, -1).cpu()
                #     cls_emb = emb[:, 0, :]
                #     layers[idx].append(cls_emb)
                features.append(image_features)
                textual_feature.append(hypo_feature)
                labels.append(target)

        features, labels, textual_feature = torch.cat(features), torch.cat(labels), torch.cat(textual_feature)
        plus_feature = textual_feature + features
        minus_feature = features - textual_feature
        times_feature = features * textual_feature

        # adapter_weights = torch.cat([cache_keys, cache_textual], dim=0)
        adapter_weights = torch.cat([features, textual_feature, plus_feature, minus_feature, times_feature], dim=-1)
        # adapter_weights = torch.cat([features, textual_feature], dim=-1)

        torch.save(adapter_weights.cpu(), cfg['cache_dir'] + "/" + split + "_feature.pt")
        torch.save(labels.cpu(), cfg['cache_dir'] + "/" + split + "_label.pt")
        # for i, emb in enumerate(layers):
        #     layers[i] = torch.cat(emb)
        #     torch.save(layers[i].cpu(), cfg['cache_dir'] + '/test_' + str(i) + "_cache.pt")
    else:
        adapter_weights = torch.load(cfg['cache_dir'] + "/" + split + "_feature.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_label.pt")
        # layers = torch.load(cfg['cache_dir'] + '/test_' + str(layer_num-1) + "_cache.pt")
        # layers = []
        # for i in range(layer_num):
        #     layer = torch.load(cfg['cache_dir'] + '/test_' + str(i) + "_cache.pt")
        #     layers.append(layer.to(device_id))
    return adapter_weights.to(device_id), labels.to(device_id)
    # return features.to(device_id), labels.to(device_id), layers.to(device_id)


def search_hp(cfg, cache_keys, cache_values, features, labels, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        if adapter:
            affinity = adapter(features)
        else:
            affinity = features @ cache_keys

        # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

        for beta in beta_list:
            # for alpha in alpha_list:
            cache_logits = affinity @ cache_values
            acc = cls_acc(cache_logits, labels)

            if acc > best_acc:
                print("New best setting, beta: {:.2f}, accuracy: {:.2f}".format(beta, acc))
                best_acc = acc
                best_beta = beta
                # best_alpha = alpha

    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta
