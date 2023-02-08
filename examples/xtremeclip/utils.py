from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    device_id = clip_model.positional_embedding.device
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda(device_id)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda(device_id)
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    device_id = clip_model.positional_embedding.device

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        # layers = {}
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []
                # cache_layers = {}
                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda(device_id)
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    # for idx, emb in layers_output.items():
                    #     emb = emb.permute(1,0,-1)
                    #     cls_emb = emb[:, 0, :]
                    #     cat_emb = []
                    #     bs = cls_emb.size(0)
                    #     for i in range(0, bs, 16):
                    #         cat_emb.append(cls_emb[i: i+16].mean(dim = 0))
                        # cache_layers[idx] = torch.stack(cat_emb, dim = 0) if idx not in cache_layers.keys() else torch.cat([cache_layers[idx], torch.stack(cat_emb, dim = 0)])

                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                # for idx, emb in cache_layers.items():
                #     layers[idx] = emb.unsqueeze(0) if idx not in layers.keys() else torch.cat([layers[idx], emb.unsqueeze(0)])

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        # cat_emb = []
        # for i in range(0, cache_keys.size(1), 16):
        #     cat_emb.append(cache_keys[:, i: i+16].mean(dim=1).unsqueeze(0))
        # cache_keys = torch.cat(cat_emb, dim=0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        # for i, emb in layers.items():
        #     layers[i] = emb.mean(dim = 0)
        #     torch.save(layers[i].cpu(), cfg['cache_dir'] + '/layer_' + str(i) + "_cache.pt")
        torch.save(cache_keys.cpu(), cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values.cpu(), cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        # layer_num = len(clip_model.visual.transformer.resblocks)
        # layers = []
        # for i in range(layer_num):
        #     layer = torch.load(cfg['cache_dir'] + '/layer_' + str(i) + "_cache.pt")
        #     layers.append(layer.to(device_id))
        # layers = torch.load(cfg['cache_dir'] + '/layer_' + str(layer_num - 1) + "_cache.pt").to(device_id)
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys.to(device_id), cache_values.to(device_id)#, layers.to(device_id)


def pre_load_features(cfg, split, clip_model, loader):

    # layer_num = len(clip_model.visual.transformer.resblocks)
    device_id = clip_model.positional_embedding.device

    if cfg['load_pre_feat'] == False:
        features, labels, layers = [], [], []
        # layers = [[] for _ in range(layer_num)]
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(device_id), target.cuda(device_id)
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # for idx, emb in image_caches.items():
                #     emb /= emb.norm(dim=-1, keepdim=True)
                #     emb = emb.permute(1, 0, -1).cpu()
                #     cls_emb = emb[:, 0, :]
                #     layers[idx].append(cls_emb)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features.cpu(), cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels.cpu(), cfg['cache_dir'] + "/" + split + "_l.pt")
        # for i, emb in enumerate(layers):
        #     layers[i] = torch.cat(emb)
        #     torch.save(layers[i].cpu(), cfg['cache_dir'] + '/test_' + str(i) + "_cache.pt")
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
        # layers = torch.load(cfg['cache_dir'] + '/test_' + str(layer_num-1) + "_cache.pt")
        # layers = []
        # for i in range(layer_num):
        #     layer = torch.load(cfg['cache_dir'] + '/test_' + str(i) + "_cache.pt")
        #     layers.append(layer.to(device_id))
    
    return features.to(device_id), labels.to(device_id)#, layers.to(device_id)


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:

        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in
                     range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in
                      range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha