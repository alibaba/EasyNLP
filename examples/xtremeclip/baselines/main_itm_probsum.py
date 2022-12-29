import os
import random
import argparse
import yaml
from tqdm import tqdm
from io import BytesIO
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from datasets.imagenet import ImageNet
import clip
from utils_itm import *
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import re
import glob
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
IMAGE_ROOT = '/home/moming/data/flickr/'
EVAL_SAVE_DIR = '/home/moming/data/flickr/itmv5/'
FLICKER_ANN_FILE = '/home/moming/data/finetune/flickr30k_%s.json'
image_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform=image_transform, image_root=IMAGE_ROOT, max_words=40):
        self.ann = []
        self.ann = json.load(open(ann_file, 'r'))
        self.ann = self.ann[:10000]
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform=image_transform, image_root=IMAGE_ROOT, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.ann = self.ann
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/moming/code/VITCLIP/configs/itm.yaml', dest='config',
                        help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def run_tip_adapter(clip_model, img_head, txt_head, train_loader, test_loader, max_epoch = 20, lr=5e-7):
    device_id = clip_model.positional_embedding.device

    optimizer = torch.optim.AdamW(
        [
            {"params": img_head.parameters(), "lr": 2e-3},
            {"params": txt_head.parameters(), "lr": 2e-3},
        ],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch * len(train_loader))
    temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) # nn.Parameter(torch.ones([]) * 0.07)

    for epoch in range(0, max_epoch):
        clip_model.eval()
        img_head.train()
        txt_head.train()
        for i, (image, text, idx) in enumerate(train_loader):
            with torch.no_grad():
                image_features, hypo_feature = encode_img_txt(image, text, clip_model, device_id)

            match_loss = matching_loss(image_features, hypo_feature, img_head, txt_head, idx, device_id, temp)
            # #### m         # loss = contra_loss + match_lossatching loss

            optimizer.zero_grad()
            match_loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 200 == 0 and i != 0:
                print("loss:" + str(match_loss.item()))
                # print("matching loss:" + str(match_loss.item()))
            #     print("learning rate:"+ str(optimizer.param_groups[0]["lr"]))

        ### do evaluation
        clip_model.eval()
        img_head.eval()
        txt_head.eval()
        with torch.no_grad():
            texts = test_loader.dataset.text
            num_text = len(texts)
            text_bs = 64
            text_embeds = []
            dump_idx = 0
            for i in range(0, num_text, text_bs):
                text = texts[i: min(num_text, i + text_bs)]
                text_embed = encode_img_txt(None, text, clip_model, device_id)
                text_embeds.append(text_embed.cpu())
                if len(text_embeds) % 16 == 0 and len(text_embeds) != 0 or i+64>num_text:
                    torch.save(torch.cat(text_embeds, dim=0), EVAL_SAVE_DIR + "test_txt_" + str(dump_idx) + ".pt")
                    text_embeds = []
                    dump_idx += 1

            image_embeds = []
            dump_img_idx = 0
            for image, img_id in test_loader:
                image_embed = encode_img_txt(image, None, clip_model, device_id)
                image_embeds.append(image_embed.cpu())
                if len(image_embeds) % 8 == 0 and len(image_embeds) != 0 or dump_img_idx * 8 * 64 + 64 > num_text:
                    torch.save(torch.cat(image_embeds, dim=0), EVAL_SAVE_DIR + "test_img_" + str(dump_img_idx) + ".pt")
                    image_embeds = []
                    dump_img_idx += 1

            base_dir = EVAL_SAVE_DIR + "test_txt_" + "%s" + ".pt"
            text_embeds = []
            for n in range(dump_idx):
                text_embeds.append(torch.load(base_dir % str(n)))
            text_embeds = torch.cat(text_embeds, dim=0).to(device_id)

            base_dir = EVAL_SAVE_DIR + "test_img_" + "%s" + ".pt"
            image_embeds = []
            for n in range(dump_img_idx):
                image_embeds.append(torch.load(base_dir % str(n)))
            image_embeds = torch.cat(image_embeds, dim=0).to(device_id)

            sims_matrix = image_embeds @ text_embeds.t() * 100
            clip_result = itm_eval(sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy(), test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(clip_result)
            adapt_logits = img_head(image_embeds) @ txt_head(text_embeds).t()
            adapt_result = itm_eval(sims_matrix.cpu().numpy() + adapt_logits.cpu().numpy(), sims_matrix.t().cpu().numpy() + adapt_logits.t().cpu().numpy(),  test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(adapt_result)
            # score_matrix_i2t = torch.full((len(test_loader.dataset.image), len(texts)), -100.0).to(device_id)
            # topk=10
            # for i, sims in enumerate(sims_matrix):
            #     topk_sim, topk_idx = sims.topk(k=topk, dim=0)
            #     img_feat = image_embeds[i].repeat(topk, 1)
            #     txt_feat = text_embeds[topk_idx]
            #     output = torch.cat([txt_feat, img_feat], dim=-1)
            #     score = img_head(output.float())
            #     score = down_head(score)[:, 1]
            #     score_matrix_i2t[i, topk_idx] = score
            #
            # sims_matrix = sims_matrix.t()
            # score_matrix_t2i = torch.full((len(texts), len(test_loader.dataset.image)), -100.0).to(device_id)
            # for i, sims in enumerate(sims_matrix):
            #     topk_sim, topk_idx = sims.topk(k=topk, dim=0)
            #     img_feat = image_embeds[topk_idx]
            #     txt_feat = text_embeds[i].repeat(topk, 1)
            #     output = torch.cat([txt_feat, img_feat], dim=-1)
            #     score = img_head(output.float())
            #     score = down_head(score)[:, 1]
            #     score_matrix_t2i[i, topk_idx] = score
            #
            # score_test_i2t, score_test_t2i = score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
            # result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            # print(result)

    # Search Hyperparameters
    # _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels)
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

def encode_img_txt(image, text, clip_model, device_id):
    if image is not None:
        image = image.to(device_id)
        image_features = clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    if text is not None:
        hypo_tensor = clip.tokenize(text).to(device_id)
        hypo_feature = clip_model.encode_text(hypo_tensor)
        hypo_feature = hypo_feature / hypo_feature.norm(dim=-1, keepdim=True)

    if image is not None and text is not None:
        return image_features, hypo_feature
    elif image is not None:
        return image_features
    elif text is not None:
        return hypo_feature

def matching_loss(image_features, hypo_feature, img_head, txt_head, idx, device_id, temp):
    bs = image_features.size(0)
    with torch.no_grad():
        sim_i2t = image_features @ hypo_feature.t() * 100
        sim_t2i = hypo_feature @ image_features.t() * 100
    adapt_i2t = img_head(image_features) @ txt_head(hypo_feature).t() * temp.exp()
    adapt_t2i = txt_head(hypo_feature) @ img_head(image_features).t() * temp.exp()

    idx = idx.view(-1, 1).to(sim_i2t.device)
    pos_idx = torch.eq(idx, idx.t())
    labels = pos_idx / pos_idx.sum(1, keepdim=True)

    contra_i2t_logits = adapt_i2t
    i2t_pos_scores = contra_i2t_logits.gather(-1, torch.range(0, idx.size(0) - 1).long().unsqueeze(-1).to(idx.device))
    ones = torch.ones(i2t_pos_scores.size()).cuda(i2t_pos_scores.device)
    loss_func = torch.nn.MarginRankingLoss(0.0, reduction='sum')
    i2t_contr_loss = loss_func(i2t_pos_scores.repeat(1, contra_i2t_logits.size(1)), contra_i2t_logits, ones)

    contra_t2i_logits = contra_i2t_logits.t()
    t2i_pos_scores = contra_t2i_logits.gather(-1, torch.range(0, idx.size(0) - 1).long().unsqueeze(-1).to(idx.device))
    t2i_contr_loss = loss_func(t2i_pos_scores.repeat(1, contra_t2i_logits.size(1)), contra_t2i_logits, ones)

    loss_i2t = -torch.sum(F.log_softmax(sim_i2t + adapt_i2t, dim=1) * labels, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i + adapt_t2i, dim=1) * labels, dim=1).mean()

    matching_loss = (loss_i2t + loss_t2i) / 2 +  i2t_contr_loss + t2i_contr_loss

    return matching_loss

def contrastive_loss(image_features, hypo_feature, idx, temp):
    logits = image_features @ hypo_feature.t() * temp.exp()
    # t2i_logits = hypo_feature @ image_features.t() * temp.exp()
    idx = idx.view(-1, 1).to(logits.device)
    pos_idx = torch.eq(idx, idx.t())
    labels = pos_idx / pos_idx.sum(1, keepdim=True)
    # ground_truth = torch.arange(idx.size(0)).long()
    # loss_i2t = i2tLoss(logits, ground_truth.to(logits.device))
    # loss_t2i = t2iLoss(t2i_logits, ground_truth.to(logits.device))
    contra_logits = image_features @ hypo_feature.t()
    contra_i2t_logits = F.log_softmax(contra_logits, dim=1)
    i2t_pos_scores = contra_i2t_logits.gather(-1, torch.range(0, idx.size(0) - 1).long().unsqueeze(-1).to(idx.device))
    ones = torch.ones(i2t_pos_scores.size()).cuda(i2t_pos_scores.device)
    loss_func = torch.nn.MarginRankingLoss(0.0, reduction='sum')
    i2t_contr_loss = loss_func(i2t_pos_scores.repeat(1, contra_i2t_logits.size(1)), contra_i2t_logits, ones)

    contra_t2i_logits = F.log_softmax(contra_logits.t(), dim=1)
    t2i_pos_scores = contra_t2i_logits.gather(-1, torch.range(0, idx.size(0) - 1).long().unsqueeze(-1).to(idx.device))
    t2i_contr_loss = loss_func(t2i_pos_scores.repeat(1, contra_t2i_logits.size(1)), contra_t2i_logits, ones)

    loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

    contrastive_loss = (loss_i2t + loss_t2i) / 2 + i2t_contr_loss + t2i_contr_loss
    return contrastive_loss

def fusion_feat(hypo_feature, image_features):
    plus_feature = hypo_feature + image_features
    minus_feature = image_features - hypo_feature
    times_feature = image_features * hypo_feature
    fusion_weights = torch.cat([image_features, hypo_feature, plus_feature, minus_feature, times_feature], dim=-1)
    return fusion_weights
# def fusion_feat(hypo_feature, image_features):
#     fusion_weights = torch.stack([hypo_feature, image_features]).mean(dim=0)
#     return fusion_weights

def load_data_file(img_data_file, txt_data_file):
    img_train_path = img_data_file % "val"
    txt_train_path = txt_data_file % "val"
    img_test_path = img_data_file % "test"
    txt_test_path = txt_data_file % "test"
    img_train_lines = [json.loads(l) for l in open(img_train_path, 'r').readlines()]
    txt_train_lines = open(txt_train_path, 'r').readlines()
    img_test_lines = [json.loads(l) for l in open(img_test_path, 'r').readlines()]
    txt_test_lines = open(txt_test_path, 'r').readlines()
    return img_train_lines, txt_train_lines, img_test_lines, txt_test_lines

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
    clip_model, preprocess = clip.load(cfg['backbone'], "cuda:5")
    clip_model.eval()
    embed_dim = 512
    itm_img = nn.Linear(embed_dim, embed_dim, bias=False).half()
    itm_img.to(clip_model.positional_embedding.device)
    itm_txt = nn.Linear(embed_dim, embed_dim, bias=False).half()
    itm_txt.to(clip_model.positional_embedding.device)


    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)

    print("Preparing ITM dataset.")
    test_dataset = re_eval_dataset(FLICKER_ANN_FILE % "test")
    train_dataset = re_train_dataset(FLICKER_ANN_FILE % "train")

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, num_workers=8, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, num_workers=8, shuffle=True)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(clip_model, itm_img, itm_txt, train_loader, test_loader)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    # run_tip_adapter_F(cfg, img_test_features, txt_test_feature, test_labels, cache_txt, clip_model, train_loader_F)

if __name__ == '__main__':
    main()