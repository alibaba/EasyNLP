#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import shutil
import copy
import time
import codecs
import datetime
import logging
import numpy as np
import argparse

from pyrouge import Rouge155

_PYROUGE_PATH = './ROUGE-1.5.5/'
_PYROUGE_TEMP_PATH = '.'

class Mapping():
    def __init__(self):
        self._char2num = {}
        self.cnt = 0

    def tonum(self, x):
        if x not in self._char2num.keys():
            self._char2num[x] = self.cnt
            self.cnt += 1
        return str(self._char2num[x])

    def charRemap(self, strlist):
        numlist = []
        for sent in strlist.split("\n"):
            numlist.append(" ".join([self.tonum(x) for x in sent.split(" ")]))
        return "\n".join(numlist)


def pyrouge_score(hypo_list, refer_list, language, level='char', convert=True, debug=False):
    """ calculate prouge for hypo and single refer

    :param hypo_list: list, each item is a (tokenized) string. Multiple sentences must be concatenated with '\n' for ROUGE-L
    :param refer_list: list for a single reference or list(list) for multiple references, the same format with hypo_list
    :param language: 'zh' and 'ja' will be split and mapped to numbers
    :param level: 'char' or 'word', only work for language='zh'. 'char' will split strings by chinese character and keep english words and numbers the same
    :return:
        scores: dict
            rouge-1: p, r, f
            rouge-2: p, r, f
            rouge-l: p, r, f
    """
    if isinstance(refer_list[0], str):    # single reference
        refer_list = [[refer] for refer in refer_list]
    assert len(hypo_list) == len(refer_list)

    if (language == 'zh' or language == 'ja' or language == 'ko') and level == 'char':
        hypo_list = [str2char(ins, language) for ins in hypo_list]
        refer_list = [[str2char(refer, language) for refer in ins] for ins in refer_list]

    if debug:
        for h, r in zip(hypo_list, refer_list):
            print('{}\t{}'.format(h, r))

    nowTime=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_PATH, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT,'hypothesis')
    MODEL_PATH = os.path.join(PYROUGE_ROOT,'reference')
    os.makedirs(SYSTEM_PATH)
    os.makedirs(MODEL_PATH)

    #r = Rouge155(rouge_dir=_PYROUGE_PATH, log_level=logging.WARNING)
    r = Rouge155(rouge_dir=_PYROUGE_PATH)
    # r = Rouge155(rouge_dir=_PYROUGE_PATH)
    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    for i in range(len(hypo_list)):
        hypo_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        # if language == 'zh' or language == 'ja' or language == 'ko':
        if language != 'en':
            mapdict = Mapping()
            refer = [mapdict.charRemap(refer_list[i][j]) for j in range(len(refer_list[i]))]
            hypo = mapdict.charRemap(hypo_list[i])
        else:
            refer = refer_list[i]
            hypo = hypo_list[i]
        with open(hypo_file, 'wb') as f:
            f.write(hypo.encode('utf-8'))

        for j in range(len(refer_list[i])):
            refer_file = os.path.join(MODEL_PATH, "Reference.%s.%d.txt" % (chr(ord('A')+j), i))
            with open(refer_file, 'wb') as f:
                f.write(refer[j].encode('utf-8'))

    try:
        output = r.convert_and_evaluate(rouge_args="-e %s/data -a -m -n 2 -d" % (_PYROUGE_PATH))
        output_dict = r.output_to_dict(output)
    finally:
        # pass
        if os.path.isdir(PYROUGE_ROOT):
            shutil.rmtree(PYROUGE_ROOT)

    scores = convertFormat(output_dict) if convert else output_dict
    return scores


################# tools #################

def splitChars(sent, lang):
    if lang == 'zh':
        parts = re.split(u"([\u4e00-\u9fa5])", sent)
    elif lang == 'ja':
        parts = re.split(u"([\u0800-\u4e00])",sent)
    elif lang == 'ko':
        parts = re.split(u"([\uac00-\ud7ff])", sent)
    else:   # Chinese, Japanese and Korean non-symbol characters
        parts = re.split(u"([\u2e80-\u9fff])", sent)
    return [p.strip().lower() for p in parts if p != "" and p != " "]

def str2char(string, language='all'):
    sents = string.split("\n")
    tokens = [" ".join(splitChars(s, language)) for s in sents]
    return "\n".join(tokens)

def convertFormat(output_dict):
    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    fullname={'p': 'precision', 'r': 'recall', 'f': 'f_score'}
    for t in ['1', '2', 'l']:
        for m in ['p', 'r', 'f']:
            scores['rouge-%s' % t][m] = output_dict['rouge_%s_%s' % (t, fullname[m])] * 100
    return scores

def rouge_results_to_str(results_dict):
    return "ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-P(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100,
        results_dict["rouge_1_precision"] * 100,
        results_dict["rouge_2_precision"] * 100,
        results_dict["rouge_l_precision"] * 100
    )

################# main #################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt", help='candidate file')
    parser.add_argument('-r', type=str, default="reference.txt", help='reference file')
    parser.add_argument('-l', type=str, default="en", help='language')
    parser.add_argument('-d', type=str, default="", help='delimiter')
    parser.add_argument('-t', action='store_true', help='need to tokenize the original document')
    parser.add_argument('-v', action='store_true', help='print detailed information')
    args = parser.parse_args()
    print(args)
    candidates = codecs.open(args.c, encoding="utf-8")
    references = codecs.open(args.r, encoding="utf-8")

    try:
        from pysbd import Segmenter
        seg = Segmenter(language=args.l, clean=False)
    except ImportError:
        raise ImportError('Please install pySBD splitor with: pip install pysbd')
    except ValueError:
        print("Unknown language code. Use language=en for segmentation.")
        seg = Segmenter(language='en', clean=False)

    print("Split sentences by pySBD\t\t" + time.strftime('%H:%M:%S', time.localtime()))

    references = [line.strip().lower() for line in references]
    candidates = [line.strip().lower() for line in candidates]
    if args.d:
        references = ["\n".join(line.split(args.d)) if args.d in line else "\n".join(seg.segment(line)) for line in references]
        candidates = ["\n".join(line.split(args.d)) if args.d in line else "\n".join(seg.segment(line)) for line in candidates]
    else:
        references = ["\n".join(seg.segment(e)) for e in references]
        candidates = ["\n".join(seg.segment(e)) for e in candidates]

    if args.t:
        try:
            from sacremoses import MosesTokenizer
            print("Tokenize string by sacremoses\t\t" + time.strftime('%H:%M:%S', time.localtime()))
            tok = MosesTokenizer(lang=args.l)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip install sacremoses')
        doc = [tok.tokenize(e, args.l, return_str=True) for e in candidates]
        summ = [tok.tokenize(e, args.l, return_str=True) for e in references]

    print("candidate: %d, reference: %d\t%s" % (len(candidates), len(references), time.strftime('%H:%M:%S', time.localtime())))

    assert len(candidates) == len(references)
    results_dict = pyrouge_score(candidates, references, args.l, convert=False, debug=args.v)
    print(rouge_results_to_str(results_dict))
    print(time.strftime('%H:%M:%S', time.localtime()))
