import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def extract_files(pos_file, neg_file, domain):
    with open(pos_file, 'r', encoding='utf-8') as f:
        pos = f.readlines()
    with open(neg_file, 'r', encoding='utf-8') as f:
        neg = f.readlines()

    corpus_pos = []
    flag = False

    for i in range(len(pos)):
        if (pos[i] == "<review_text>\n"):
            flag = True
            new_str = " "
            continue
        elif (pos[i] == "</review_text>\n"):
            flag = False
            corpus_pos.append(new_str)
            continue
        if (flag):
            if (len(pos[i]) > 1):
                sent = pos[i]
                sent = sent[0:len(sent) - 1]
                if (sent[0] == '\t'):
                    sent = sent[1:len(sent) - 1]
            new_str += sent

    corpus_neg = []
    flag = False

    for i in range(len(neg)):
        if (neg[i] == "<review_text>\n"):
            flag = True
            new_str = " "
            continue
        elif (neg[i] == "</review_text>\n"):
            flag = False
            corpus_neg.append(new_str)
            continue
        if (flag):
            if (len(neg[i]) > 1):
                sent = neg[i]
                sent = sent[0:len(sent) - 1]
                if (sent[0] == '\t'):
                    sent = sent[1:len(sent) - 1]
            new_str += sent

    data = list()

    for corpus in corpus_pos:
        data.append({
            "review": corpus.replace("\t", " "),
            "domain": domain,
            "sentiment": "positive"})

    for corpus in corpus_neg:
        data.append({
            "review": corpus.replace("\t", " "),
            "domain": domain,
            "sentiment": "negative"})

    return data

if __name__ == "__main__":
    all_data = list()
    for domain in ["books", "dvd", "electronics", "kitchen_&_housewares"]:
        data = extract_files("./data/sorted_data_acl/%s/positive.review" % domain,
                             "./data/sorted_data_acl/%s/negative.review" % domain,
                             domain.split("_")[0])
        all_data.extend(data)
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1.0, random_state=11111)

    skf = StratifiedKFold(n_splits=10, random_state=11111, shuffle=True)
    for tr_idx, te_idx in skf.split(df, df["sentiment"]):
        raw_tr_df = df.iloc[tr_idx]
        te_df = df.iloc[te_idx]
        break

    skf = StratifiedKFold(n_splits=10, random_state=11111, shuffle=True)
    for tr_idx, val_idx in skf.split(raw_tr_df, raw_tr_df["sentiment"]):
        tr_df = raw_tr_df.iloc[tr_idx]
        val_df = raw_tr_df.iloc[val_idx]
        break

    if not os.path.exists("data/SENTI"):
        os.makedirs("data/SENTI/")
    tr_df.to_csv("./data/SENTI/train.tsv", sep="\t", index=False)
    val_df.to_csv("./data/SENTI/dev.tsv", sep="\t", index=False)
    te_df.to_csv("./data/SENTI/test.tsv", sep="\t", index=False)

    print("Train distribution: ")
    print(tr_df.domain.value_counts())
    print("Valid distribution: ")
    print(val_df.domain.value_counts())
    print("Test distribution: ")
    print(te_df.domain.value_counts())



