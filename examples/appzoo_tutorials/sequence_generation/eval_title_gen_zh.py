import sys
from rouge import Rouge
from collections import defaultdict

if __name__ == "__main__":
    pred_titles = list()
    grt_titles = list()
    pred_dict = defaultdict(list)
    grt_dict = defaultdict(list)
    with open(sys.argv[1]) as f:
        for line in f:
            pred,candi, grt, _, tag = line.strip().split("\t")
            pred = " ".join(pred.replace(" ", ""))
            grt = " ".join(grt.replace(" ", ""))
            pred_titles.append(pred)
            grt_titles.append(grt)
            pred_dict[tag].append(pred)
            grt_dict[tag].append(grt)

    print("=" * 10 + " Overall " + "=" * 10)
    rouge = Rouge()
    scores = rouge.get_scores(pred_titles, grt_titles, avg=True)
    print("Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f}".format(
        scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100, scores['rouge-l']['f'] * 100))

    for tag in pred_dict:
        print("=" * 10 + " %s " % tag + "=" * 10)
        scores = rouge.get_scores(pred_dict[tag], grt_dict[tag], avg=True)
        print("Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f}".format(
            scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100, scores['rouge-l']['f'] * 100))