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
import pandas as pd

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
    parser.add_argument(
        '--image-standard-path', 
        type=str, 
        required=True,
        help="Specify the path of image ground-truth file."
    )
    parser.add_argument(
        '--text-standard-path', 
        type=str, 
        required=True,
        help="Specify the path of text ground-truth file."
    )       
    parser.add_argument(
        '--image-out-path', 
        type=str, 
        required=True,
        help="Specify the image output json filepath."
    )
    parser.add_argument(
        '--text-out-path', 
        type=str, 
        required=True,
        help="Specify the text output json filepath."
    )                   
    return parser.parse_args()


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)


def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)


def report_score(r1, r5, r10, out_p):
    result = dict()
    result['success']=True
    mean_recall = (r1 + r5 + r10) / 3.0
    result['score'] = mean_recall * 100
    result['scoreJson'] = {'score': mean_recall * 100, 'mean_recall': mean_recall * 100, 'r1': r1 * 100, 'r5': r5 * 100, 'r10': r10 * 100}
    dump_2_json(result,out_p)


def read_reference(path, key="query_id"):
    fin = open(path)
    reference = dict()
    for line in fin:
        line = line.strip()
        obj = json.loads(line)
        if 'item_ids' in obj:
            reference[obj[key]] = obj['item_ids']
        else:
            reference[obj[key]] = [obj['text']]
    return reference

def compute_score(query_key, golden_file, predictions, output_path):
    # read ground-truth
    reference = read_reference(golden_file, query_key)

    # compute score for each query
    r1_stat, r5_stat, r10_stat = 0, 0, 0
    for qid in reference.keys():
        ground_truth_ids = set(reference[qid])
        top10_pred_ids = predictions[qid]
        if any([idx in top10_pred_ids[:1] for idx in ground_truth_ids]):
            r1_stat += 1
        if any([idx in top10_pred_ids[:5] for idx in ground_truth_ids]):
            r5_stat += 1
        if any([idx in top10_pred_ids[:10] for idx in ground_truth_ids]):
            r10_stat += 1
    # the higher score, the better
    r1, r5, r10 = r1_stat * 1.0 / len(reference), r5_stat * 1.0 / len(reference), r10_stat * 1.0 / len(reference)
    mean_recall = (r1 + r5 + r10) / 3.0
    result = [mean_recall, r1, r5, r10]
    result = [item * 100 for item in result]

    report_score(r1, r5, r10, output_path)

    return result

def evaluation(query_key, standard_path, submit_path, out_path):
    try:
        evaluation_result = compute_score(query_key, standard_path, submit_path, out_path)
        print("The evaluation is saved in {}".format(out_path))
    except Exception as e:
        report_error_msg(e.args[0], e.args[0], out_path)
        print("The evaluation failed: {}".format(e.args[0]))


if __name__ == "__main__":
    args = parse_args()

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
            if "query_id" not in obj:
                text_ids.append(obj['image_id'])
            else:
                text_ids.append(obj['query_id'])
            text_feats.append(obj['feature'])
    text_feats_array = np.array(text_feats, dtype=np.float32)
    print("Finished loading text features.")

    text_predictions = {}
    print("Begin to compute top-{} predictions for queries...".format(args.top_k))
    with open(args.output_texts, "w") as fout:
        with open(args.text_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())

                query_id = obj['query_id']
                text_feat = obj['feature']

                score_tuples = []
                text_feat_tensor = torch.tensor([text_feat], dtype=torch.float).cuda() # [1, feature_dim]
                idx = 0
                while idx < len(image_ids):
                    img_feats_tensor = torch.from_numpy(image_feats_array[idx : min(idx + args.eval_batch_size, len(image_ids))]).cuda() # [batch_size, feature_dim]
                    batch_scores = text_feat_tensor @ img_feats_tensor.t() # [1, batch_size]
                    for image_id, score in zip(image_ids[idx : min(idx + args.eval_batch_size, len(image_ids))], batch_scores.squeeze(0).tolist()):
                        score_tuples.append((image_id, score))
                    idx += args.eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:args.top_k]
                fout.write("{}\n".format(json.dumps({"query_id": query_id, "item_ids": [entry[0] for entry in top_k_predictions]})))

                text_predictions[query_id] = [entry[0] for entry in top_k_predictions]

    print("Top-{} predictions are saved in {}".format(args.top_k, args.output_texts))
    print("Done!")

    image_predictions = {}
    print("Begin to compute top-{} predictions for images...".format(args.top_k))
    with open(args.output_images, "w") as fout:
        with open(args.image_feats, "r") as fin:
            for line in tqdm(fin):
                obj = json.loads(line.strip())

                image_id = obj['image_id']
                image_feat = obj['feature']

                score_tuples = []
                image_feat_tensor = torch.tensor([image_feat], dtype=torch.float).cuda() # [1, feature_dim]
                idx = 0
                while idx < len(text_ids):
                    text_feats_tensor = torch.from_numpy(text_feats_array[idx : min(idx + args.eval_batch_size, len(text_ids))]).cuda() # [batch_size, feature_dim]
                    batch_scores = image_feat_tensor @ text_feats_tensor.t() # [1, batch_size]
                    for text_id, score in zip(text_ids[idx : min(idx + args.eval_batch_size, len(text_ids))], batch_scores.squeeze(0).tolist()):
                        score_tuples.append((text_id, score))
                    idx += args.eval_batch_size
                top_k_predictions = sorted(score_tuples, key=lambda x:x[1], reverse=True)[:args.top_k]
                fout.write("{}\n".format(json.dumps({"image_id": image_id, "item_ids": [entry[0] for entry in top_k_predictions]})))

                image_predictions[image_id] = [entry[0] for entry in top_k_predictions]

        print("Top-{} predictions are saved in {}".format(args.top_k, args.output_images))
        print("Done!")

    print("Begin to evaluate images and texts...")

    evaluation("query_id", args.text_standard_path, text_predictions, args.text_out_path)
    evaluation("image_id", args.image_standard_path, image_predictions, args.image_out_path)

    print("Done evaluation!")