import sys
from collections import defaultdict
from edit_distance import SequenceMatcher
from rouge import Rouge

if __name__ == "__main__":
    pred_titles = list()
    grt_titles = list()
    pred_dict = defaultdict(list)
    grt_dict = defaultdict(list)
    ed_len={}
    ed_cnt=0
    rouge_sum={'1':0,'2':0,'l':0}
    rouge_cnt={'1':0,'2':0,'l':0}
    
    with open(sys.argv[1]) as f:
        for line in f:
            pred,cans, grt, _ = line.strip().split("\t")
            pred_titles.append(pred)
            grt_titles.append(grt)

    print("=" * 10 + " Overall " + "=" * 10)
    rouge = Rouge()
    scores = rouge.get_scores(pred_titles, grt_titles, avg=True)
    print("Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f}".format(
        scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100, scores['rouge-l']['f'] * 100))