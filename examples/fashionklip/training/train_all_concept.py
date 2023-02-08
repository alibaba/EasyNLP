import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

import sys
import pdb
import wandb

import logging

from tqdm import tqdm

import faiss

from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def mmr(query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        idxes: List[str],
        top_n: int = 10,
        diversity: float = 0.8) -> List[str]:

    query_candidate_similarity = cosine_similarity(candidate_embeddings, query_embedding)
    candidate_similarity = cosine_similarity(candidate_embeddings)

    keywords_idx = [np.argmax(query_candidate_similarity)]
    candidates_idx = [i for i in range(len(idxes)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = query_candidate_similarity[candidates_idx, :]
        target_similarities = np.max(candidate_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [idxes[idx] for idx in keywords_idx]

def load_image_data(args, model, image_data_sets):
    dataloader = image_data_sets.dataloader
    image_feat_dict = {}
    m = model.module if args.distributed or args.dp else model
    m.eval()
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_ids, images = batch
            images = images.cuda(args.gpu, non_blocking=True)
            image_features = m(images, None)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            for image_id, image_feature in zip(image_ids, image_features.tolist()):
                image_feat_dict[image_id] = image_feature
                cnt += 1

    print("Finished forwarding image features with {} items.".format(str(cnt)))
    
    return image_feat_dict


def load_concept_data(args, model, concept_data):
    dataloader = concept_data.dataloader
    concept_feat_dict = {}
    m = model.module if args.distributed or args.dp else model
    m.eval()
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            query_ids, texts = batch
            texts = texts.cuda(args.gpu, non_blocking=True)
            text_features = m(None, texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            for query_id, text_feature in zip(query_ids.tolist(), text_features.tolist()):
                concept_feat_dict[query_id] = text_feature
                cnt += 1
        print('Finished forwarding concept features with {} items'.format(str(cnt)))
    
    return concept_feat_dict 


def faiss_prediction(dim, concept_features, image_features):
    concept_prediction_dict = {}

    concept_ids = []
    concept_feats = []
    for concept_id, concept_feat in tqdm(concept_features.items()):
        concept_ids.append(concept_id)
        concept_feats.append(concept_feat)
    
    concept_feats = np.array(concept_feats[:], dtype=np.float32)

    image_ids = []
    image_feats = []
    for image_id, image_feat in tqdm(image_features.items()):
        image_ids.append(int(image_id))
        image_feats.append(image_feat)
    
    image_feats = np.array(image_feats[:], dtype=np.float32)
    image_ids = np.array(image_ids[:])

    nlist, k = 1, 20
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss.normalize_L2(image_feats)
    index.train(image_feats)
    index.add_with_ids(image_feats, image_ids)
    faiss.normalize_L2(concept_feats)             
    distances, indexes = index.search(concept_feats, k)

    for concept_id, prediction in zip(concept_ids, indexes):
        prediction = [str(item) for item in prediction]
        pred_feats = []
        for pred_id in prediction:
            pred_feats.append(image_features[pred_id])
        pred_feats = np.array(pred_feats[:])
        concept_id_feat = np.array([concept_features[concept_id]])
        diverse_preds = mmr(concept_id_feat, pred_feats, prediction)
        diverse_preds = [int(pred_item) for pred_item in diverse_preds]
        concept_prediction_dict[concept_id] = diverse_preds

    return concept_prediction_dict


def is_master(args):
    return (not args.distributed) or args.gpu == 0

def get_loss(model, images, texts, concepts, concepts_images_features, loss_img, loss_txt, args):
    image_features, text_features, logit_scale = model(images, texts)

    concepts_features = []
    bs, con_len, context_length = concepts.shape
    bs_len, feat_len = image_features.shape
    for concept_idx in range(con_len):
        text_concepts = [concept[concept_idx].unsqueeze(0) for concept in concepts]
        text_concepts = torch.cat(text_concepts)

        concept_features = model.module(None, text_concepts)
        concept_features = concept_features / concept_features.norm(p=2, dim=-1, keepdim=True)
        concepts_features.append(concept_features)

    logit_scale = logit_scale.mean()
    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        gathered_all_concepts_features = []
        for concept_features in concepts_features:
            gathered_concept_features = [torch.zeros_like(concept_features) for _ in range(world_size)]
            dist.all_gather(gathered_concept_features, concept_features)
            
            all_concept_features = torch.cat([concept_features] + gathered_concept_features[:rank] + gathered_concept_features[rank + 1 :])
            gathered_all_concepts_features.append(all_concept_features)
        
        gathered_all_concepts_images_features = [
            torch.zeros_like(concepts_images_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_all_concepts_images_features, concepts_images_features)
        all_concepts_images_features = torch.cat(
            [concepts_images_features]
            + gathered_all_concepts_images_features[:rank]
            + gathered_all_concepts_images_features[rank + 1 :]
        )

        
        # print(all_concepts_images_features.shape)
        # expected torch.Size([64, 3, 10, 512])
        
        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        total_bs, img_feat_len = all_image_features.shape

        all_image_features_resize = all_image_features.unsqueeze(1)
        all_image_features_resize = all_image_features_resize.permute(0, 2, 1)

        # print(all_image_features_resize.shape)
        # expected torch.Size([64, 512, 1])

        all_image_con_concepts_images_index = []
        all_image_con_concepts_weights = []
        for con_idx in range(con_len):
            con_concepts_images_features = all_concepts_images_features[:, con_idx, :, :].squeeze(dim=1)

            image_con_concepts_images = torch.bmm(con_concepts_images_features, all_image_features_resize).permute(0, 2, 1)

            image_con_concepts_images_index_values, image_con_concepts_images_index = image_con_concepts_images.topk(k=5, dim=2)

            image_con_concepts_images_index = image_con_concepts_images_index.squeeze(1)
            image_con_concepts_images_index_values = image_con_concepts_images_index_values.squeeze(1)

            max_concepts_images_features = []
            for topk_idx in range(5):
                image_con_concepts_images_index_values[:, topk_idx] = image_con_concepts_images_index_values[:, topk_idx] / image_con_concepts_images_index_values[:, topk_idx].norm(p=2, dim=-1, keepdim=True)
                max_concepts_images_features_idx = torch.cat([con_concepts_images_features[bs_idx, max_idx, :].unsqueeze(0) for bs_idx, max_idx in enumerate(image_con_concepts_images_index[:, topk_idx].tolist())])
                max_concepts_images_features_idx = max_concepts_images_features_idx.unsqueeze(dim=1)

                max_concepts_images_features.append(max_concepts_images_features_idx)
            
            max_concepts_images_features = torch.cat(max_concepts_images_features, dim=1)

            all_image_con_concepts_images_index.append(max_concepts_images_features)
            all_image_con_concepts_weights.append(image_con_concepts_images_index_values)
        
        logits_concept_per_image = []
        logits_per_concept = []
        for topk_idx in range(5):
            logits_concept_per_image_idx = [logit_scale * all_image_con_concepts_images_index[concept_idx][:, topk_idx, :] @ gathered_all_concepts_features[concept_idx].t() for concept_idx in range(con_len)]
            logits_per_concept_idx = [logits_concept_image.t() for logits_concept_image in logits_concept_per_image_idx]

            logits_concept_per_image.append(logits_concept_per_image_idx)
            logits_per_concept.append(logits_per_concept_idx)

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2

    concept_loss = 0
    for concept_idx in range(con_len):
        concept_idx_loss = 0
        concept_weight_sum = sum(sum(all_image_con_concepts_weights[concept_idx]))
        for topk_idx in range(5):
            concept_idx_weight = sum(all_image_con_concepts_weights[concept_idx][:, topk_idx]) / concept_weight_sum
            concept_idx_weight = concept_idx_weight.detach()

            logits_concept_per_image_idx = logits_concept_per_image[concept_idx][topk_idx].cuda(args.gpu)
            logits_per_concept_idx = logits_per_concept[concept_idx][topk_idx].cuda(args.gpu)

            idx_weight = all_image_con_concepts_weights[concept_idx][:, topk_idx]

            loss_con_img = nn.CrossEntropyLoss(weight=idx_weight.detach())
            loss_con_txt = nn.CrossEntropyLoss(weight=idx_weight.detach())

            if args.gpu is not None:
                loss_con_img = loss_con_img.cuda(args.gpu)
                loss_con_txt = loss_con_txt.cuda(args.gpu)

            concept_idx_loss += concept_idx_weight * (loss_con_img(logits_concept_per_image_idx, ground_truth) + loss_con_txt(logits_per_concept_idx, ground_truth)) / 2
        
        concept_loss += concept_idx_loss
    
    concept_loss /= con_len

    return total_loss, concept_loss


def predict_concept_images(args, ids, predictions, concept_image_feats, feat_dim):
    bs = len(ids[0])
    con_len = len(ids)

    preds_resize = []
    for idx in range(bs):
        batch_preds = []
        for j in range(con_len):
            if ids[j][idx] != -1:
                concepts_preds = predictions[int(ids[j][idx].item())]
            else:
                concepts_preds = [""] * 10
            batch_preds.append(concepts_preds)
        preds_resize.append(batch_preds)

    all_concept_images = []
    for pred in preds_resize:
        concept_images = []
        for pred_item in pred:
            pred_features = [torch.from_numpy(np.array(concept_image_feats[str(pred_id)], dtype=np.float32)).cuda(args.gpu).unsqueeze(0) 
                                if pred_id != "" else torch.Tensor(np.array([0] * feat_dim, dtype=np.float32)).unsqueeze(0).cuda(args.gpu, non_blocking=True) 
                                for pred_id in pred_item]
            pred_features = torch.cat(pred_features)
            concept_images.append(pred_features.unsqueeze(0).cuda(args.gpu, non_blocking=True))
        concept_images = torch.cat(concept_images).cuda(args.gpu, non_blocking=True)

        all_concept_images.append(concept_images.unsqueeze(0))
    
    all_concept_images = torch.cat(all_concept_images).cuda(args.gpu, non_blocking=True)
    # print(all_concept_images.shape)

    return all_concept_images


def gather_text_feat_dict(text_feature_dict, rank, world_size, args):
    text_feature_dict_keys = torch.Tensor(list(text_feature_dict.keys())).cuda(args.gpu)
    text_feature_dict_values = torch.Tensor(list(text_feature_dict.values())).cuda(args.gpu)

    gathered_text_feature_dict_keys = [
        torch.zeros_like(text_feature_dict_keys) for _ in range(world_size)
    ]
    gathered_text_feature_dict_values = [
        torch.zeros_like(text_feature_dict_values) for _ in range(world_size)
    ]
    dist.all_gather(gathered_text_feature_dict_keys, text_feature_dict_keys)
    dist.all_gather(gathered_text_feature_dict_values, text_feature_dict_values)

    all_text_feature_dict_keys = torch.cat(
        [text_feature_dict_keys]
        + gathered_text_feature_dict_keys[:rank]
        + gathered_text_feature_dict_keys[rank + 1 :]
    )
    all_text_feature_dict_values = torch.cat(
        [text_feature_dict_values]
        + gathered_text_feature_dict_values[:rank]
        + gathered_text_feature_dict_values[rank + 1 :]
    )

    all_text_feature_dict_keys = all_text_feature_dict_keys.cpu().numpy().tolist()
    all_text_feature_dict_values = all_text_feature_dict_values.cpu().numpy().tolist()

    all_text_feature_dict = {}
    for key_idx, feat_key_id in tqdm(enumerate(all_text_feature_dict_keys)):
        feat_value = all_text_feature_dict_values[key_idx]
        all_text_feature_dict[feat_key_id] = feat_value
    
    return all_text_feature_dict


def gather_image_feat_dict(image_feature_dict, rank, world_size, args):
    image_feature_dict_keys = [int(pred_id) for pred_id in list(image_feature_dict.keys())]

    image_feature_dict_keys = torch.Tensor(image_feature_dict_keys).cuda(args.gpu)

    image_feature_dict_values = torch.Tensor(list(image_feature_dict.values())).cuda(args.gpu)

    gathered_image_feature_dict_keys = [
        torch.zeros_like(image_feature_dict_keys) for _ in range(world_size)
    ]
    gathered_image_feature_dict_values = [
        torch.zeros_like(image_feature_dict_values) for _ in range(world_size)
    ]
    dist.all_gather(gathered_image_feature_dict_keys, image_feature_dict_keys)
    dist.all_gather(gathered_image_feature_dict_values, image_feature_dict_values)

    all_image_feature_dict_keys = torch.cat(
        [image_feature_dict_keys]
        + gathered_image_feature_dict_keys[:rank]
        + gathered_image_feature_dict_keys[rank + 1 :]
    )
    all_image_feature_dict_values = torch.cat(
        [image_feature_dict_values]
        + gathered_image_feature_dict_values[:rank]
        + gathered_image_feature_dict_values[rank + 1 :]
    )

    all_image_feature_dict_keys = all_image_feature_dict_keys.cpu().numpy().tolist()
    all_image_feature_dict_values = all_image_feature_dict_values.cpu().numpy().tolist()

    all_image_feature_dict = {}
    for key_idx, feat_key_id in tqdm(enumerate(all_image_feature_dict_keys)):
        feat_value = all_image_feature_dict_values[key_idx]
        all_image_feature_dict[str(int(feat_key_id))] = feat_value
        
    return all_image_feature_dict

def gather_prediction_dict(concept_image_predictions, rank, world_size, args):
    concept_image_predictions_keys = torch.Tensor(list(concept_image_predictions.keys())).cuda(args.gpu)
    concept_image_predictions_values = torch.Tensor(list(concept_image_predictions.values())).cuda(args.gpu)

    gathered_concept_image_predictions_keys = [
        torch.zeros_like(concept_image_predictions_keys) for _ in range(world_size)
    ]
    gathered_concept_image_predictions_values = [
        torch.zeros_like(concept_image_predictions_values) for _ in range(world_size)
    ]
    dist.all_gather(gathered_concept_image_predictions_keys, concept_image_predictions_keys)
    dist.all_gather(gathered_concept_image_predictions_values, concept_image_predictions_values)

    all_concept_image_predictions_keys = torch.cat(
        [concept_image_predictions_keys]
        + gathered_concept_image_predictions_keys[:rank]
        + gathered_concept_image_predictions_keys[rank + 1 :]
    )
    all_concept_image_predictions_values = torch.cat(
        [concept_image_predictions_values]
        + gathered_concept_image_predictions_values[:rank]
        + gathered_concept_image_predictions_values[rank + 1 :]
    )

    all_concept_image_predictions_keys = all_concept_image_predictions_keys.cpu().numpy().tolist()
    all_concept_image_predictions_values = all_concept_image_predictions_values.cpu().numpy().tolist()

    all_concept_image_predictions = {}
    for key_idx, concept_key_id in tqdm(enumerate(all_concept_image_predictions_keys)):
        concept_image_preds = all_concept_image_predictions_values[key_idx]
        concept_image_preds = [str(int(concept_image_pred_item)) for concept_image_pred_item in concept_image_preds]
        all_concept_image_predictions[concept_key_id] = concept_image_preds
    
    print("All-gather concept-image prediction keys:", len(all_concept_image_predictions.keys()))

    return all_concept_image_predictions

def train(model, data, image_data_sets, concept_data, kb_concept_data, feat_dim, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    print("Training starts...")
    # os.environ["WDS_EPOCH"] = str(epoch)

    concept_image_feats = load_image_data(args, model, image_data_sets)
    print("Concept images length:", len(concept_image_feats.values()))

    concept_feats = load_concept_data(args, model, concept_data)
    kb_concept_feats = load_concept_data(args, model, kb_concept_data)
    print("Concepts length:", len(concept_feats.values()))

    concept_image_predictions = faiss_prediction(feat_dim, concept_feats, concept_image_feats)
    kb_concept_image_predictions = faiss_prediction(feat_dim, kb_concept_feats, concept_image_feats)

    if args.distributed and args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        concept_feats = gather_text_feat_dict(concept_feats, rank, world_size, args)
        kb_concept_feats = gather_text_feat_dict(concept_feats, rank, world_size, args)

        print("All-gather concepts length:", len(concept_feats.keys()))
        print("All-gather KB concepts length:", len(kb_concept_feats.keys()))

        concept_image_feats = gather_image_feat_dict(concept_image_feats, rank, world_size, args)
        print("All-gather concept images length:", len(concept_image_feats.keys()))

        concept_image_predictions = gather_prediction_dict(concept_image_predictions, rank, world_size, args)
        kb_concept_image_predictions = gather_prediction_dict(kb_concept_image_predictions, rank, world_size, args)
        
        print("All-gather concept-image prediction keys:", len(concept_image_predictions.keys()))
        print("All-gather KB concept-image prediction keys:", len(kb_concept_image_predictions.keys()))

    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    print(f"batches: {num_batches_per_epoch}, dataloader: {len(dataloader)}")

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, texts, concepts, kb_concepts, ids, kb_ids = batch

        all_data_concept_images = predict_concept_images(args, ids, concept_image_predictions, concept_image_feats, feat_dim)
        all_kb_concept_images = predict_concept_images(args, kb_ids, kb_concept_image_predictions, concept_image_feats, feat_dim)

        all_concept_images = torch.cat([all_data_concept_images, all_kb_concept_images], dim=1)

        all_concepts = torch.cat([concepts, kb_concepts], dim=1)
        print("All concepts length:", all_concepts.shape)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)
            all_concepts = concepts.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                contrast_loss, concept_loss = get_loss(model, images, texts, all_concepts, all_concept_images, loss_img, loss_txt, args)

                concept_weight = (concept_loss / contrast_loss).detach()
                concept_weight = 1 / (5 * concept_weight) if concept_weight != 0.0 else 0.0

                contrast_weight = 1.0 - concept_weight

                total_loss = contrast_weight * contrast_loss + concept_weight * concept_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            contrast_loss, concept_loss = get_loss(model, images, texts, concepts, all_concept_images, loss_img, loss_txt, args)

            concept_weight = (concept_loss / contrast_loss).detach()
            concept_weight = 1 / (5 * concept_weight) if concept_weight != 0.0 else 0.0

            contrast_weight = 1.0 - concept_weight
                                
            total_loss = contrast_weight * contrast_loss + concept_weight * concept_loss

            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        if is_master(args) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch

            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\t Contrastive_Loss: {contrast_loss.item():.6f}\t Concept_Loss: {concept_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "contrastive_loss": contrast_loss.item(),
                "concept_loss": concept_loss.item(),
                "data_time": data_time,
                "batch_time": batch_time,
                "scale":  m.logit_scale.data.item(),
                "lr": optimizer.param_groups[0]["lr"]
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, timestep)
                if args.wandb:
                    wandb.log({name: val, 'step': timestep})


def evaluate(model, data, epoch, args, tb_writer=None, steps=None):
    if not is_master(args):
        return
    
    logging.info(f"Begin to eval epoch: {epoch}...")
    print((f"Begin to eval epoch: {epoch}..."))
    
    model.eval()

    dataloader = data['val'].dataloader

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

    cumulative_loss = 0.0
    num_elements = 0.0
    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, texts = batch
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                texts = texts.cuda(args.gpu, non_blocking=True)

            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            if args.gpu is not None:
                ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size
        
        metrics = {}
        loss = cumulative_loss / num_elements
        metrics.update(
            **{"val_loss": loss.item(), "epoch": epoch, "num_elements": num_elements}
        )

        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        )

    return metrics