# -*- coding: utf-8 -*-
'''
This script computes the recall scores given the ground-truth annotations and predictions.
'''

import json
import sys
import os
import string
import numpy as np
import time

from tqdm import tqdm

NUM_K = 10

def read_submission(submit_path, reference, k=5, keyword="query_id"):
    # check whether the path of submitted file exists
    if not os.path.exists(submit_path):
        raise Exception("The submission file is not found!")

    submission_dict = {}
    ref_qids = set(reference.keys())

    with open(submit_path) as fin:
        for line in fin:
            line = line.strip()
            try:
                pred_obj = json.loads(line)
            except:
                raise Exception('Cannot parse this line into json object: {}'.format(line))
            if keyword not in pred_obj:
                continue
            qid = pred_obj[keyword]
            if "item_ids" not in pred_obj:
                raise Exception('There exists one line not containing the predicted item_ids: {}'.format(line))
            item_ids = pred_obj["item_ids"]
            # print(len(item_ids))
            if isinstance(item_ids[0], list):
                item_ids = [item[0] for item in item_ids]
            if not isinstance(item_ids, list):
                raise Exception('The item_ids field of query_id {} is not a list, please check your schema'.format(qid))
            submission_dict[qid] = item_ids # here we save the list of product ids
    return submission_dict


def read_reference(path, keyword="query_id"):
    fin = open(path)
    reference = dict()
    for line in fin:
        line = line.strip()
        obj = json.loads(line)
        reference[obj[keyword]] = obj['item_ids']
    return reference

def try_rerank(predictions, back_predictions):
    rerank_predictions = {}
    for qid, preds in predictions.items():
        top1_pred = preds[0]
        back_top1_pred = back_predictions[top1_pred]
        if qid in back_top1_pred and back_top1_pred.index(qid) == 0:
            rerank_predictions[qid] = preds
        else:
            rank_score = []
            for pred_rank, pred_id in enumerate(preds):
                back_preds = back_predictions[pred_id]
                if qid not in back_preds:
                    back_rank = 10000
                else:
                    back_rank = back_preds.index(qid)
                avg_rank = (pred_rank + back_rank) / 2
                rank_score.append((pred_id, avg_rank))
            rank_score.sort(key=lambda k: k[1], reverse=False)
            rerank_predictions[qid] = [score_item[0] for score_item in rank_score]
    
    return rerank_predictions

def compute_score(reference, predictions):
    r1_stat, r5_stat, r10_stat = 0, 0, 0

    for qid in reference.keys():
        ground_truth_ids = set(reference[qid])

        if not isinstance(list(predictions.keys())[0], type(qid)):
            top10_pred_ids = predictions[str(qid)]
        else:
            top10_pred_ids = predictions[qid]
        # print(ground_truth_ids, top10_pred_ids)
        if not isinstance(top10_pred_ids[0], type(list(ground_truth_ids)[0])):
            top10_pred_ids = [int(idx) for idx in top10_pred_ids]

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
    print(result)
    return result

def save_rerank_results(rerank_results, rerank_save_path, key):
    with open(rerank_save_path, "w") as fout:
        for rerank_id, pred_items in rerank_results.items():
            fout.write("{}\n".format(json.dumps({key: rerank_id, "item_ids": pred_items})))

if __name__=="__main__":
    image_standard_path = sys.argv[1]
    image_submit_path = sys.argv[2]
    text_standard_path = sys.argv[3]
    text_submit_path = sys.argv[4]

    image_rerank_save_path = sys.argv[5]
    text_rerank_save_path = sys.argv[6]

    # read ground-truth
    image_reference = read_reference(image_standard_path, "image_id")
    text_reference = read_reference(text_standard_path, "query_id")
    
    # read predictions
    image_predictions = read_submission(image_submit_path, image_reference, 10, "image_id")
    text_predictions = read_submission(text_submit_path, text_reference, 10, "query_id")

    image_rerank_predictions = try_rerank(image_predictions, text_predictions)
    text_rerank_predictions = try_rerank(text_predictions, image_predictions)

    save_rerank_results(image_rerank_predictions, image_rerank_save_path, "image_id")
    save_rerank_results(text_rerank_predictions, text_rerank_save_path, "query_id")

    compute_score(text_reference, text_predictions)
    compute_score(text_reference, text_rerank_predictions)

    compute_score(image_reference, image_predictions)
    compute_score(image_reference, image_rerank_predictions)
        