# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''

import os
import argparse
import logging
from pathlib import Path
import json

import torch
from tqdm import tqdm

from clip.model import convert_weights, CLIP, build_model
from training.main_all import convert_models_to_fp32
from eval.data import get_eval_img_dataset, get_eval_txt_dataset, get_eval_mask_dataset, get_eval_prompt_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--extract-image-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract image features."
    )
    parser.add_argument(
        '--extract-text-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract text features."
    )
    parser.add_argument(
        '--extract-mask-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract prompt mask features."
    )
    parser.add_argument(
        '--extract-prompt-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract prompt features."
    )
    parser.add_argument(
        '--image-data', 
        type=str, 
        default="", 
        help="If --extract-image-feats is True, specify the path of processed image npzfile."
    )
    parser.add_argument(
        '--text-data', 
        type=str, 
        default="", 
        help="If --extract-text-feats is True, specify the path of test query jsonl file."
    )
    parser.add_argument(
        '--prompt-data', 
        type=str, 
        default="", 
        help="If --extract-prompt-feats is True, specify the path of test prompt jsonl file."
    )
    parser.add_argument(
        '--image-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output image features."
    )    
    parser.add_argument(
        '--text-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output text features."
    )
    parser.add_argument(
        '--prompt-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-prompt-feats is True, specify the path of output prompt features."
    )
    parser.add_argument(
        '--mask-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-mask-feats is True, specify the path of output prompt masks features."
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=64, help="Text batch size."
    )    
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--model",
        choices=["ViT-B-32", "ViT-L-14"],
        default="ViT-L-14",
        help="Name of the vision backbone to use.",
    )    
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )   
    args = parser.parse_args()

    return args    


if __name__ == "__main__":
    args = parse_args()

    assert args.extract_image_feats or args.extract_text_feats, "--extract-image-feats and --extract-text-feats cannot both be False!"

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")
    
    args.gpu = 0
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
    model_config_file = Path(__file__).parent / f"../training/model_configs/{args.model.replace('/', '-')}.json"
    print('Loading model from', model_config_file)
    assert os.path.exists(model_config_file)
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info)
    convert_weights(model)

    # Get data.
    if args.extract_image_feats:
        print("Preparing image inference dataset.")
        img_data = get_eval_img_dataset(args)
    if args.extract_text_feats:
        print("Preparing text inference dataset.")
        text_data = get_eval_txt_dataset(args, max_txt_length=model_info['context_length'])
    if args.extract_prompt_feats:
        print("Preparing prompt inference dataset.")
        prompt_data = get_eval_prompt_dataset(args, max_txt_length=model_info['context_length'])
    if args.extract_mask_feats:
        print("Preparing mask inference dataset.")
        mask_data = get_eval_mask_dataset(args, max_txt_length=model_info['context_length'])
    
    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(args.gpu)

    # zero-shot load
    checkpoint = torch.jit.load(args.resume, map_location='cpu')
    sd = checkpoint.state_dict()
    model = build_model(sd).cuda(args.gpu)

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

    print(
        f"=> loaded checkpoint '{args.resume}'"
    )

    # Make inference for images
    if args.extract_image_feats:
        print('Make inference for images...')
        if args.image_feat_output_path is None:
            args.image_feat_output_path = "{}.img_feat.jsonl".format(args.image_data[:-4])
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            model.eval()
            dataloader = img_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    image_ids, images = batch
                    images = images.cuda(args.gpu, non_blocking=True)
                    image_features = model(images, None)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids, image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))


    # Make inference for texts
    if args.extract_text_feats:
        print('Make inference for texts...')
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            model.eval()
            dataloader = text_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    query_ids, texts = batch
                    texts = texts.cuda(args.gpu, non_blocking=True)
                    text_features = model(None, texts)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    for query_id, text_feature in zip(query_ids.tolist(), text_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"query_id": query_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} query text features are stored in {}'.format(write_cnt, args.text_feat_output_path))
    
    print("Done!")

    # Make inference for prompts
    if args.extract_prompt_feats:
        print('Make inference for prompts...')
        if args.prompt_feat_output_path is None:
            args.prompt_feat_output_path = "{}.prompt_feat.jsonl".format(args.prompt_data[:-6])
        write_cnt = 0
        with open(args.prompt_feat_output_path, "w") as fout:
            model.eval()
            dataloader = prompt_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    query_ids, texts = batch
                    texts = texts.cuda(args.gpu, non_blocking=True)
                    # print(texts.shape)
                    texts_features = []
                    for prompt in texts:
                        prompt_features = model(None, prompt)
                        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
                        texts_features.append(prompt_features.tolist())

                    for query_id, text_feature in zip(query_ids.tolist(), texts_features):
                        fout.write("{}\n".format(json.dumps({"query_id": query_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} query prompt features are stored in {}'.format(write_cnt, args.text_feat_output_path))
    
    print("Done!")

     # Make inference for mask prompts
    if args.extract_mask_feats:
        print('Make inference for prompts...')
        if args.mask_feat_output_path is None:
            args.mask_feat_output_path = "{}.mask_feat.jsonl".format(args.mask_data[:-6])
        write_cnt = 0
        with open(args.mask_feat_output_path, "w") as fout:
            model.eval()
            dataloader = mask_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    query_ids, texts = batch
                    texts = texts.cuda(args.gpu, non_blocking=True)
                    # print(texts.shape)
                    texts_features = []
                    for mask in texts:
                        mask_features = model(None, mask)
                        mask_features /= mask_features.norm(dim=-1, keepdim=True)
                        texts_features.append(mask_features.tolist())

                    for query_id, text_feature in zip(query_ids.tolist(), texts_features):
                        fout.write("{}\n".format(json.dumps({"query_id": query_id, "feature": text_feature})))
                        write_cnt += 1
        print('{} query prompt mask features are stored in {}'.format(write_cnt, args.text_feat_output_path))
    
    print("Done!")