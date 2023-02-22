import os
import time
import json
import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import autocast
import torch.distributed as dist

from .data import parse_obj_dict

import sys
import pdb
import wandb

import logging

from tqdm import tqdm
import random

from clip.clip import tokenize

def is_master(args):
    return (not args.distributed) or args.gpu == 0

def get_loss(model, images, texts, prompts, masks, da_images, da_images_masks, da_vg_images_masks, da_texts, loss_img, loss_txt, loss_mask, args):
    image_features, text_features, logit_scale = model(images, texts)

    da_image_features = model.module(da_images, None)
    da_image_mask_features = model.module(da_images_masks, None)
    da_text_features = model.module(None, da_texts)
    da_vg_image_mask_features = model.module(da_vg_images_masks, None)

    da_image_features = da_image_features / da_image_features.norm(dim=-1, keepdim=True)
    da_text_features = da_text_features / da_text_features.norm(dim=-1, keepdim=True)
    da_image_mask_features = da_image_mask_features / da_image_mask_features.norm(dim=-1, keepdim=True)
    da_vg_image_mask_features = da_vg_image_mask_features / da_vg_image_mask_features.norm(dim=-1, keepdim=True)

    logit_scale = logit_scale.mean()

    all_prompts_features = []
    bs, prompt_len, context_length = prompts.shape
    for prompt_index in range(prompt_len):
        text_prompts = [text_prompt[prompt_index].unsqueeze(0) for text_prompt in prompts]
        text_prompts = torch.cat(text_prompts)

        prompts_features = model.module(None, text_prompts)
        prompts_features = prompts_features / prompts_features.norm(dim=-1, keepdim=True)
        all_prompts_features.append(prompts_features)
    
    all_masks_features = []
    bs, prompt_len, context_length = masks.shape
    for prompt_index in range(prompt_len):
        text_masks = [text_mask[prompt_index].unsqueeze(0) for text_mask in masks]
        text_masks = torch.cat(text_masks)

        masks_features = model.module(None, text_masks)
        masks_features = masks_features / masks_features.norm(dim=-1, keepdim=True)
        all_masks_features.append(masks_features)

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

        gathered_da_image_features = [
            torch.zeros_like(da_image_features) for _ in range(world_size)
        ]
        gathered_da_text_features = [
            torch.zeros_like(da_text_features) for _ in range(world_size)
        ]
        gathered_da_image_mask_features = [
            torch.zeros_like(da_image_mask_features) for _ in range(world_size)
        ]

        gathered_da_vg_image_mask_features = [
            torch.zeros_like(da_vg_image_mask_features) for _ in range(world_size)
        ]

        dist.all_gather(gathered_da_image_features, da_image_features)
        dist.all_gather(gathered_da_image_mask_features, da_image_mask_features)
        dist.all_gather(gathered_da_text_features, da_text_features)
        dist.all_gather(gathered_da_vg_image_mask_features, da_vg_image_mask_features)

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

        all_da_image_features = torch.cat(
            [da_image_features]
            + gathered_da_image_features[:rank]
            + gathered_da_image_features[rank + 1 :]
        )
        all_da_text_features = torch.cat(
            [da_text_features]
            + gathered_da_text_features[:rank]
            + gathered_da_text_features[rank + 1 :]
        )
        all_da_image_mask_features = torch.cat(
            [da_image_mask_features]
            + gathered_da_image_mask_features[:rank]
            + gathered_da_image_mask_features[rank + 1 :]
        )

        all_da_vg_image_mask_features = torch.cat(
            [da_vg_image_mask_features]
            + gathered_da_vg_image_mask_features[:rank]
            + gathered_da_vg_image_mask_features[rank + 1 :]
        )

        gathered_all_prompts_features = []
        for prompt_features in all_prompts_features:
            gathered_prompt_features = [torch.zeros_like(prompt_features) for _ in range(world_size)]
            dist.all_gather(gathered_prompt_features, prompt_features)
            
            all_prompt_features = torch.cat([prompt_features] + gathered_prompt_features[:rank] + gathered_prompt_features[rank + 1 :])
            gathered_all_prompts_features.append(all_prompt_features)
        
        gathered_all_masks_features = []
        for mask_features in all_masks_features:
            gathered_mask_features = [torch.zeros_like(mask_features) for _ in range(world_size)]
            dist.all_gather(gathered_mask_features, mask_features)
            
            all_mask_features = torch.cat([mask_features] + gathered_mask_features[:rank] + gathered_mask_features[rank + 1 :])
            gathered_all_masks_features.append(all_mask_features)

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        logits_per_da_image = logit_scale * all_da_image_features @ all_da_text_features.t()
        logits_per_da_text = logits_per_da_image.t()
        logits_per_da_mask_image = logit_scale * all_da_image_mask_features @ all_da_text_features.t()

        logits_per_image_prompts = [logit_scale * all_image_features @ prompts_features.t() for prompts_features in gathered_all_prompts_features]
        logits_per_prompts = [logits_per_image_prompt.t() for logits_per_image_prompt in logits_per_image_prompts]

        logits_per_image_masks = [logit_scale * all_image_features @ masks_features.t() for masks_features in gathered_all_masks_features]

        logits_per_da_vg_mask_image = logit_scale * all_da_vg_image_mask_features @ all_text_features.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        logits_per_da_image = logit_scale * da_image_features @ da_text_features.t()
        logits_per_da_text = logit_scale * da_text_features @ da_image_features.t()
        logits_per_da_mask_image = logit_scale * da_image_mask_features @ da_text_features.t()

        logits_per_image_prompts = [logit_scale * image_features @ prompts_features.t() for prompts_features in all_prompts_features]
        logits_per_prompts = [logit_scale * prompts_features @ image_features.t() for prompts_features in all_prompts_features]

        logits_per_image_masks = [logit_scale * image_features @ masks_features.t() for masks_features in all_masks_features]

        logits_per_da_vg_mask_image = logit_scale * da_vg_image_mask_features @ text_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    da_ground_truth = torch.arange(len(logits_per_da_image)).long()

    x, y = logits_per_da_mask_image.shape
    total_bs = x

    target = torch.tensor([1]*total_bs)
    
    logits_per_image_gt = [logits_per_image[row_index][row_index] for row_index in range(total_bs)]
    logits_per_da_image_gt = [logits_per_da_image[row_index][row_index] for row_index in range(total_bs)]
    logits_per_da_mask_image = [logits_per_da_mask_image[row_index][row_index] for row_index in range(total_bs)]

    logits_per_image_gt = torch.tensor(logits_per_image_gt)
    logits_per_da_image_gt = torch.tensor(logits_per_da_image_gt)
    logits_per_da_mask_image = torch.tensor(logits_per_da_mask_image)

    logits_per_da_vg_mask_image = [logits_per_da_vg_mask_image[row_index][row_index] for row_index in range(total_bs)]
    logits_per_da_vg_mask_image = torch.tensor(logits_per_da_vg_mask_image)

    if args.gpu is not None:
        ground_truth = ground_truth.cuda(args.gpu, non_blocking=True)
        da_ground_truth = da_ground_truth.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        logits_per_image_gt = logits_per_image_gt.cuda(args.gpu, non_blocking=True)
        logits_per_da_image_gt = logits_per_da_image_gt.cuda(args.gpu, non_blocking=True)
        logits_per_da_mask_image = logits_per_da_mask_image.cuda(args.gpu, non_blocking=True)
        logits_per_da_vg_mask_image = logits_per_da_vg_mask_image.cuda(args.gpu, non_blocking=True)
    
    da_mask_loss = loss_mask(logits_per_da_image_gt, logits_per_da_mask_image, target)
    da_mask_loss /= logit_scale

    da_vg_mask_loss = loss_mask(logits_per_image_gt, logits_per_da_vg_mask_image, target)
    da_vg_mask_loss /= logit_scale

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2

    da_total_loss = (
        loss_img(logits_per_da_image, da_ground_truth)
        + loss_txt(logits_per_da_text, da_ground_truth)
    ) / 2

    prompt_loss = 0
    for prompt_index in range(prompt_len):
        logits_per_image_prompt = logits_per_image_prompts[prompt_index]
        logits_per_prompt = logits_per_prompts[prompt_index]

        prompt_loss += (loss_img(logits_per_image_prompt, ground_truth) + loss_txt(logits_per_prompt, ground_truth)) / 2
    
    prompt_loss /= prompt_len

    mask_loss = 0
    for prompt_index in range(prompt_len):
        logits_per_image_mask = logits_per_image_masks[prompt_index]
        
        logits_per_image_mask_gt = [logits_per_image_mask[row_index][row_index] for row_index in range(total_bs)]
        logits_per_image_mask_gt = torch.tensor(logits_per_image_mask_gt)

        if args.gpu is not None:
            logits_per_image_mask_gt = logits_per_image_mask_gt.cuda(args.gpu, non_blocking=True)

        mask_loss += loss_mask(logits_per_image_gt, logits_per_image_mask_gt, target)
    
    mask_loss /= logit_scale

    return total_loss, prompt_loss, mask_loss, da_total_loss, da_mask_loss, da_vg_mask_loss

def train(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    print("Training starts...")
    # os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_mask = nn.MarginRankingLoss(reduction="sum")

    if args.gpu is not None:
        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)

        loss_mask = loss_mask.cuda(args.gpu)

    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches

    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images, da_images, da_texts, da_images_masks, da_vg_images_masks, texts, texts_prompts, texts_masks = batch

        da_texts = da_texts.flatten(0, 1)
        da_images = da_images.flatten(0, 1)
        da_images_masks = da_images_masks.flatten(0, 1)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            texts = texts.cuda(args.gpu, non_blocking=True)

            da_images = da_images.cuda(args.gpu, non_blocking=True)
            da_texts = da_texts.cuda(args.gpu, non_blocking=True)
            da_images_masks = da_images_masks.cuda(args.gpu, non_blocking=True)

            da_vg_images_masks = da_vg_images_masks.cuda(args.gpu, non_blocking=True)

            texts_prompts = texts_prompts.cuda(args.gpu, non_blocking=True)
            texts_masks = texts_masks.cuda(args.gpu, non_blocking=True)

        data_time = time.time() - end

        m = model.module if args.distributed or args.dp else model

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss, prompt_loss, mask_loss, da_total_loss, da_mask_loss, vg_mask_loss = get_loss(model, images, texts, texts_prompts, texts_masks, da_images, da_images_masks, da_vg_images_masks, da_texts, loss_img, loss_txt, loss_mask, args)

                total_loss = 0.90 * total_loss + 0.03 * prompt_loss + 0.10 * mask_loss + 0.03 * da_total_loss + 0.15 * da_mask_loss + 0.10 * vg_mask_loss

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss, prompt_loss, mask_loss, da_total_loss, da_mask_loss, vg_mask_loss = get_loss(model, images, texts, texts_prompts, texts_masks, da_images, da_images_masks, da_vg_images_masks, da_texts, loss_img, loss_txt, loss_mask, args)

            total_loss = 0.90 * total_loss + 0.03 * prompt_loss + 0.10 * mask_loss + 0.03 * da_total_loss + 0.15 * da_mask_loss + 0.10 * vg_mask_loss

            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        # if is_master(args) and (i % 100) == 0:
        if is_master(args) == 0:
            num_samples = i * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]\t"
                f"Loss: {total_loss.item():.6f}\t Prompt_Loss: {prompt_loss.item():.6f}\t Mask_Loss: {mask_loss.item():.6f}\t DA_Loss: {da_total_loss.item():.6f}\t DA_Mask_Loss: {da_mask_loss.item():.6f}\t VG_Mask_Loss: {vg_mask_loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                f"\tLR: {optimizer.param_groups[0]['lr']:5f}\tlogit_scale {m.logit_scale.data:.3f}"
            )
            # save train loss / etc.

            timestep = epoch * num_batches_per_epoch + i
            log_data = {
                "loss": total_loss.item(),
                "prompt_loss": prompt_loss.item(),
                "mask_loss": mask_loss.item(),
                "da_mask_loss": da_mask_loss.item(),
                "da_total_loss": da_total_loss.item(),
                "vg_mask_loss": vg_mask_loss.item(),
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
