import os
import argparse
from os.path import join, exists
import subprocess as sp
import json
import tempfile
import multiprocessing as mp
from time import time
from datetime import timedelta
import queue
import logging
from itertools import combinations

from cytoolz import curry
from pyrouge.utils import log
from pyrouge import Rouge155

from transformers import BertTokenizer, RobertaTokenizer
# from transformers import BertTokenizer

MAX_LEN = 512

_ROUGE_PATH = '/root/moming/code/SciSoft/ROUGE-1.5.5'
temp_path = '/root/moming/code/MatchSum_efl/preprocess/temp' # path to store some temporary files

original_data, sent_ids = [], []

def load_jsonl(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_rouge(path, dec):
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'
    dec_dir = join(path, 'decode')
    ref_dir = join(path, 'reference')

    with open(join(dec_dir, '0.dec'), 'w') as f:
        for sentence in dec:
            print(sentence, file=f)

    cmd = '-c 95 -r 1000 -n 2 -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
            + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
            + cmd
            + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)

        line = output.split('\n')
        rouge1 = float(line[3].split(' ')[3])
        rouge2 = float(line[7].split(' ')[3])
        rougel = float(line[11].split(' ')[3])
    return (rouge1 + rouge2 + rougel) / 3

@curry
def get_candidates(tokenizer, cls, sep_id, idx):

    idx_path = join(temp_path, str(idx))

    # create some temporary files to calculate ROUGE
    #if not os.path.exists(idx_path):
    sp.call('mkdir ' + idx_path, shell=True)
    #if not os.path.exists(join(idx_path, 'decode')):
    sp.call('mkdir ' + join(idx_path, 'decode'), shell=True)
    #if not os.path.exists(join(idx_path, 'reference')):
    sp.call('mkdir ' + join(idx_path, 'reference'), shell=True)

    # load data
    data = {}
    data['text'] = original_data[idx]['text']
    data['summary'] = original_data[idx]['summary']

    # write reference summary to temporary files
    ref_dir = join(idx_path, 'reference')
    with open(join(ref_dir, '0.ref'), 'w') as f:
        for sentence in data['summary']:
            print(sentence, file=f)

    # get candidate summaries
    # here is for CNN/DM: truncate each document into the 5 most important sentences (using BertExt),
    # then select any 2 or 3 sentences to form a candidate summary, so there are C(5,2)+C(5,3)=20 candidate summaries.
    # if you want to process other datasets, you may need to adjust these numbers according to specific situation.
    sent_id = original_data[idx]['sent_id'][:5]
    indices = list(combinations(sent_id, 2))
    indices += list(combinations(sent_id, 3))
    #indices = list(combinations(sent_id, 3))
    if len(sent_id) <1:
        indices += list(combinations(sent_id, len(sent_id)))

    # get ROUGE score for each candidate summary and sort them in descending order
    score = []
    for i in indices:
        i = list(i)
        i.sort()
        # write dec
        dec = []
        for j in i:
            sent = data['text'][j]
            dec.append(sent)
        score.append((i, get_rouge(idx_path, dec)))
    score.sort(key=lambda x : x[1], reverse=True)

    # write candidate indices and score
    data['ext_idx'] = sent_id
    data['indices'] = []
    data['score'] = []
    for i, R in score:
        data['indices'].append(list(map(int, i)))
        data['score'].append(R)

    # tokenize and get candidate_id
    candidate_summary = []
    for i in data['indices']:
        cur_summary = [cls]
        for j in i:
            cur_summary += data['text'][j].split()
        cur_summary = cur_summary[:MAX_LEN]
        cur_summary = ' '.join(cur_summary)
        candidate_summary.append(cur_summary)

    data['candidate_id'] = []
    for summary in candidate_summary:
        token_ids = tokenizer.encode(summary, add_special_tokens=False)[:(MAX_LEN - 1)]
        token_ids += sep_id
        data['candidate_id'].append(token_ids)

    # tokenize and get text_id
    text = [cls]
    for sent in data['text']:
        text += sent.split()
    text = text[:MAX_LEN]
    text = ' '.join(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:(MAX_LEN - 1)]
    token_ids += sep_id
    data['text_id'] = token_ids

    # tokenize and get summary_id
    summary = [cls]
    for sent in data['summary']:
        summary += sent.split()
    summary = summary[:MAX_LEN]
    summary = ' '.join(summary)
    token_ids = tokenizer.encode(summary, add_special_tokens=False)[:(MAX_LEN - 1)]
    token_ids += sep_id
    data['summary_id'] = token_ids

    # write processed data to temporary file
    processed_path = join(temp_path, 'processed')
    with open(join(processed_path, '{}.json'.format(idx)), 'w') as f:
        json.dump(data, f, indent=4)

    sp.call('rm -r ' + idx_path, shell=True)

def get_candidates_mp(args):
    
    # choose tokenizer
    if args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('/root/moming/pretrained_language_models/bert_eng')
        cls, sep = '[CLS]', '[SEP]'
    else:
        tokenizer = RobertaTokenizer.from_pretrained('/root/moming/pretrained_language_models/roberta')
        cls, sep = '<s>', '</s>'
    sep_id = tokenizer.encode(sep, add_special_tokens=False)

    # load original data and indices
    global original_data, sent_ids
    original_data = load_jsonl(args.data_path)
    # sent_ids = load_jsonl(args.index_path)
    n_files = len(original_data)
    # assert len(sent_ids) == len(original_data)
    print('total {} documents'.format(n_files))
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    processed_path = join(temp_path, 'processed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    # use multi-processing to get candidate summaries
    start = time()
    print('start getting candidates with multi-processing !!!')
    
    with mp.Pool(40) as pool:
        list(pool.imap_unordered(get_candidates(tokenizer, cls, sep_id), range(n_files), chunksize=64))
    
    print('finished in {}'.format(timedelta(seconds=time()-start)))
    
    # write processed data
    print('start writing {} files'.format(n_files))
    for i in range(n_files):
        with open(join(processed_path, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        with open(args.write_path, 'a') as f:
            print(json.dumps(data), file=f)
    
    os.system('rm -r {}'.format(temp_path))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Process truncated documents to obtain candidate summaries'
    )
    # parser.add_argument('--tokenizer', type=str, default='bert',#required=True,
    parser.add_argument('--tokenizer', type=str, default='roberta',#required=True,
        help='BERT/RoBERTa')
    parser.add_argument('--data_path', type=str, default='/root/moming/result/presumm/test_match.jsonl',# required=True,
        help='path to the original dataset, the original dataset should contain text and summary')
    parser.add_argument('--index_path', type=str, default='/root/moming/data/disco/index',#required=True,
        help='indices of the remaining sentences of the truncated document')
    parser.add_argument('--write_path', type=str, default='/root/moming/data/disco/bigbird_test_CNNDM_bert.jsonl', # required=True,
        help='path to store the processed dataset')

    args = parser.parse_args()
    assert args.tokenizer in ['bert', 'roberta']
    assert exists(args.data_path)
    assert exists(args.index_path)

    get_candidates_mp(args)
