# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs prediction file for evaluation.
'''

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-feats', 
        type=str, 
        required=True,
        help="Specify the path of image features."
    )  
    parser.add_argument(
        '--text-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )
    parser.add_argument(
        '--mask-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )
    parser.add_argument(
        '--prompt-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help="Specify the k value of top-k predictions."
    )   
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        default=8192,
        help="Specify the image-side batch size when computing the inner products, default to 8192"
    )    
    parser.add_argument(
        '--output-texts', 
        type=str, 
        required=True,
        help="Specify the texts output jsonl prediction filepath."
    )
    parser.add_argument(
        '--output-images', 
        type=str, 
        required=True,
        help="Specify the images output jsonl prediction filepath."
    )                
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    topk = 128

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    with open(args.image_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    print("Finished loading image features.")

    print("Begin to load text features...")
    text_ids = []
    text_feats = []
    with open(args.text_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_ids.append(obj['query_id'])
            text_feats.append(obj['feature'])
    text_feats_array = np.array(text_feats, dtype=np.float32)
    print("Finished loading text features.")

    print("Begin to load prompt features...")
    prompt_feats = {}
    prompt_feats_list = [[]*len(text_feats) for _ in range(10)]
    with open(args.prompt_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            obj_prompt_feats = obj['feature']
            prompt_feats[obj['query_id']] = obj_prompt_feats
            for prompt_index, prompt_feat in enumerate(obj_prompt_feats):
                prompt_feats_list[prompt_index].append(prompt_feat)
    prompt_feats_arrays = []
    for prompt_feats_item in prompt_feats_list:
        prompt_feats_array = np.array(prompt_feats_item, dtype=np.float32)
        prompt_feats_arrays.append(prompt_feats_array)
    print(len(prompt_feats_arrays), len(prompt_feats_arrays[0]))
    print("Finished loading prompt features.")

    print("Begin to load mask features...")
    mask_feats = {}
    mask_feats_list = [[]*len(text_feats) for _ in range(10)]
    with open(args.mask_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            obj_mask_feats = obj['feature']
            mask_feats[obj['query_id']] = obj_mask_feats
            for mask_index, mask_feat in enumerate(obj_mask_feats):
                mask_feats_list[mask_index].append(mask_feat)
    mask_feats_arrays = []
    for mask_feats_item in mask_feats_list:
        mask_feats_array = np.array(mask_feats_item, dtype=np.float32)
        mask_feats_arrays.append(mask_feats_array)
    print(len(mask_feats_arrays), len(mask_feats_arrays[0]))
    print("Finished loading mask features.")

    print("Begin to compute top-{} predictions for queries...".format(args.top_k))

    txt_cnt = 0
    with open(args.output_texts, "w") as fout:
        with open(args.text_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())
                txt_cnt += 1
                
                query_id = obj['query_id']
                text_feat = obj['feature']
                score_tuples = []
                text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).cuda() # [1, feature_dim]

                prompt_feat_list = prompt_feats[query_id]
                prompt_feat_tensors = [torch.tensor([text_feat], dtype=torch.float).cuda() for text_feat in prompt_feat_list]

                mask_feat_list = mask_feats[query_id]
                mask_feat_tensors = [torch.tensor([text_feat], dtype=torch.float).cuda() for text_feat in mask_feat_list]

                idx = 0
                while idx < len(image_ids):
                    img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).cuda() # [batch_size, feature_dim]
                    batch_scores = text_feat_tensor @ img_feats_tensor.t() # [1, batch_size]

                    masks_batch_scores = []
                    for mask_feat_tensor in mask_feat_tensors:
                        mask_batch_score  = mask_feat_tensor @ img_feats_tensor.t() # [1, batch_size]
                        masks_batch_scores.append(mask_batch_score.tolist()[0])
                    
                    masks_batch_scores = np.array(masks_batch_scores).T.tolist()

                    prompts_batch_scores = []
                    for prompt_feat_tensor in prompt_feat_tensors:
                        prompt_batch_score  = prompt_feat_tensor @ img_feats_tensor.t()
                        prompts_batch_scores.append(prompt_batch_score.tolist()[0])
                    
                    prompt_batch_scores = np.mean(np.array(prompts_batch_scores), axis=0).tolist()

                    for image_id, score, prompt_score, mask_scores in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], batch_scores.squeeze(0).tolist(), prompt_batch_scores, masks_batch_scores):
                        # print((image_id, score, mask_scores))
                        score_diffs = [(score - mask_score) for mask_score in mask_scores]
                        pos_cnt = len([diff for diff in score_diffs if diff > 0])

                        mask_score_all = sum(score_diffs)

                        overall_score = 0.9 * score + 0.1 * mask_score_all

                        score_tuples.append((image_id, score, overall_score, 0.99 * overall_score + 0.01 * prompt_score))

                    idx += args.eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:topk]
                first_rerank_top_k = sorted(top_k_predictions, key=lambda x:x[2], reverse=True)
                rerank_top_k = sorted(first_rerank_top_k, key=lambda x:x[3], reverse=True)[:args.top_k]
                fout.write("{}\n".format(json.dumps({"query_id": query_id, "item_ids": [(entry[0], entry[1], entry[2], entry[3]) for entry in rerank_top_k]})))
    
    print("Top-{} predictions are saved in {}".format(args.top_k, args.output_texts))
    print("Done!")

    print("Begin to compute top-{} predictions for images...".format(args.top_k))
    img_cnt = 0
    with open(args.output_images, "w") as fout:
        with open(args.image_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())

                img_cnt += 1

                image_id = obj['image_id']
                image_feat = obj['feature']
                score_tuples = []
                image_feat_tensor = torch.tensor([image_feat], dtype=torch.float).cuda() # [1, feature_dim]
                idx = 0
                while idx < len(text_ids):
                    text_feats_tensor = torch.from_numpy(text_feats_array[idx : min(idx + args.eval_batch_size, len(text_ids))]).cuda() # [batch_size, feature_dim]
                    batch_scores = image_feat_tensor @ text_feats_tensor.t() # [1, batch_size]

                    masks_scores = []
                    for mask_feats_item in mask_feats_arrays:
                        mask_feats_tensor = torch.from_numpy(mask_feats_item[idx : min(idx + args.eval_batch_size, len(text_ids))]).cuda()
                        mask_scores = image_feat_tensor @ mask_feats_tensor.t()
                        masks_scores.append(mask_scores.tolist()[0])
                    
                    masks_scores = np.array(masks_scores).T.tolist()

                    prompts_scores = []
                    for prompt_feats_item in prompt_feats_arrays:
                        prompt_feats_tensor = torch.from_numpy(prompt_feats_item[idx : min(idx + args.eval_batch_size, len(text_ids))]).cuda()
                        prompt_scores = image_feat_tensor @ prompt_feats_tensor.t()
                        prompts_scores.append(prompt_scores.tolist()[0])
                    
                    prompts_batch_scores = np.mean(np.array(prompts_scores), axis=0).tolist()

                    for text_id, score, prompt_score, mask_scores in zip(text_ids[idx : min(idx + args.eval_batch_size, len(text_ids))], batch_scores.squeeze(0).tolist(), prompts_batch_scores, masks_scores):
                        # print(text_id, score, mask_scores)
                        score_diffs = [min(0, score - mask_score) for mask_score in mask_scores]
                        pos_cnt = len([diff for diff in score_diffs if diff > 0])

                        overall_score = 0.8 * score + 0.2 * prompt_score
                        mask_score_all = sum(score_diffs)

                        score_tuples.append((text_id, score, overall_score, 0.98 * overall_score + 0.02 * mask_score_all))

                    idx += args.eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:topk]
                first_rerank_top_k = sorted(top_k_predictions, key=lambda x:x[2], reverse=True)
                rerank_top_k = sorted(first_rerank_top_k, key=lambda x:x[3], reverse=True)[:args.top_k]

                fout.write("{}\n".format(json.dumps({"image_id": image_id, "item_ids": [(entry[0], entry[1], entry[2], entry[3]) for entry in rerank_top_k]})))

    print("Top-{} predictions are saved in {}".format(args.top_k, args.output_images))
    print("Done!")