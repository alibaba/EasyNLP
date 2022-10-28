# -*- coding: utf-8 -*-
'''
Evaluation script for CMRC 2018
version: v5 - special
Note:
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
'''
import re
import nltk


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    return list(in_str)
# def mixed_segmentation(in_str, rm_punc=False):
#     in_str = str(in_str).lower().strip()
#     segs_out = []
#     temp_str = ""
#     sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
#                '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
#                '「', '」', '（', '）', '－', '～', '『', '』']
#     for char in in_str:
#         if rm_punc and char in sp_char:
#             continue
#         if re.search('[\u4e00-\u9fa5]', char) or char in sp_char:
#             if temp_str != "":
#                 ss = nltk.word_tokenize(temp_str)
#                 segs_out.extend(ss)
#                 temp_str = ""
#             segs_out.append(char)
#         else:
#             temp_str += char
#
#     if temp_str != "":
#         ss = nltk.word_tokenize(temp_str)
#         segs_out.extend(ss)
#
#     return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


#
def evaluate2(prediction, ground_truth):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for id1, answers in ground_truth.items():
        pred = prediction[id1]
        f1 += calc_f1_score(answers, pred)
        em += calc_em_score(answers, pred)
        total_count += 1

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {'f1': round(f1_score, 4), 'em': round(em_score, 4)}

def evaluate(prediction, ground_truth):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    prediction = {p['id']: p['prediction_text'] for p in prediction}
    for i in ground_truth:
        id1 = i['id']
        answers = i['answers']['text']
        pred = prediction[id1]
        f1 += calc_f1_score(answers, pred)
        em += calc_em_score(answers, pred)
        total_count += 1

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return {'f1': round(f1_score, 4), 'em': round(em_score, 4)}


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em
