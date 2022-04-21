'''
convert clue dataset to easynlp style inputs
'''
import json

if __name__ == "__main__":
    for file in ['train', 'dev']:
        with open('{}.json'.format(file), 'r', encoding='utf-8') as fr:
            lines = fr.readlines()

        fw = open('{}.tsv'.format(file), 'w', encoding='utf-8')
        for line in lines:
            line = json.loads(line.replace('\n', ''))
            sent1 = " ".join(line['keyword'])
            sent2 = line['abst']
            label = line['label']
            fw.write("{}\t{}\t{}\n".format(label, sent1, sent2))
        fw.close()