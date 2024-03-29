# -*- coding: utf-8 -*-
'''
This script computes the recall scores given the ground-truth annotations and predictions.
'''

import json
import sys
import os
import string
import numpy as np

NUM_K = 10

def read_submission(submit_path, reference, k=5, key="query_id"):
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
            if key not in pred_obj:
                continue
            qid = int(pred_obj[key])
            qid = pred_obj[key]
            if "item_ids" not in pred_obj:
                raise Exception('There exists one line not containing the predicted item_ids: {}'.format(line))
            item_ids = pred_obj["item_ids"]
            if isinstance(item_ids[0], list):
                item_ids = [item[0] for item in item_ids]
            if not isinstance(item_ids, list):
                raise Exception('The item_ids field of {} {} is not a list, please check your schema'.format(key, qid))

            submission_dict[int(qid)] = item_ids # here we save the list of product ids
            
    return submission_dict


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

def compute_score(golden_file, predict_file):
    # read ground-truth
    reference = read_reference(golden_file)

    # read predictions
    k = 10
    predictions = read_submission(predict_file, reference, k)
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
    return result


if __name__=="__main__":
    # the path of answer json file (eg. test_queries_answers.jsonl)
    standard_path = sys.argv[1]
    # the path of prediction file (eg. example_pred.jsonl)
    submit_path = sys.argv[2]
    # the score will be dumped into this output json file
    out_path = sys.argv[3]

    query_key = sys.argv[4]

    print("Read standard from %s" % standard_path)
    print("Read user submit file from %s" % submit_path)

    try:
        # read ground-truth
        reference = read_reference(standard_path, query_key)
        
        # read predictions
        k = 10
        predictions = read_submission(submit_path, reference, k, query_key)

        # compute score for each query
        r1_stat, r5_stat, r10_stat = 0, 0, 0
        for qid in reference.keys():
            ground_truth_ids = set(reference[qid])
            top10_pred_ids = predictions[qid]

            top10_pred_ids = [int(top_id) for top_id in top10_pred_ids]

            if not isinstance(list(ground_truth_ids)[0], type(top10_pred_ids[0])):
                ground_truth_ids = [int(idx) for idx in ground_truth_ids]

            if any([idx in top10_pred_ids[:1] for idx in ground_truth_ids]):
                r1_stat += 1
            if any([idx in top10_pred_ids[:5] for idx in ground_truth_ids]):
                r5_stat += 1
            if any([idx in top10_pred_ids[:10] for idx in ground_truth_ids]):
                r10_stat += 1
        # the higher score, the better
        r1, r5, r10 = r1_stat * 1.0 / len(reference), r5_stat * 1.0 / len(reference), r10_stat * 1.0 / len(reference)
        mean_recall = (r1 + r5 + r10) / 3.0
        print(mean_recall, r1, r5, r10, out_path)
        report_score(r1, r5, r10, out_path)
        print("The evaluation finished successfully.")
    except Exception as e:
        report_error_msg(e.args[0], e.args[0], out_path)
        print("The evaluation failed: {}".format(e.args[0]))