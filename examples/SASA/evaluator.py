# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['target']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            idx,label=line.split()
            predictions[int(idx)]=int(label)
    return predictions

def calculate_scores(answers,predictions):
    Acc=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key]==predictions[key])

    scores={}
    scores['Acc']=np.mean(Acc)
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)

if __name__ == '__main__':
    main()
