import os
import argparse

def process(args):
    names = os.listdir(os.path.join(args.result_path, 'dec'))
    names = sorted(names, key=lambda n: int(n.split('.')[0]))

    cache = []
    for n in names:
        summs = open(os.path.join(args.result_path,'dec',n), 'r').readlines()
        summs = [s.replace('\n', '') for s in summs]
        cache.append("<q>".join(summs))
    with open(os.path.join(args.rouge_result_path, "candidate.txt"), 'w') as f:
        for c in cache:
            print(c, file=f)

    names = os.listdir(os.path.join(args.result_path, 'ref'))
    names = sorted(names, key=lambda n: int(n.split('.')[0]))

    cache = []
    for n in names:
        summs = open(os.path.join(args.result_path, 'ref',n), 'r').readlines()
        summs = [s.replace('\n', '') for s in summs]
        cache.append("<q>".join(summs))
    with open(os.path.join(args.rouge_result_path, "reference.txt"), 'w') as f:
        for c in cache:
            print(c, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of MatchSum'
    )
    parser.add_argument('--result_path', default='', required=True,
                        help='training or testing of MatchSum', type=str)
    parser.add_argument('--rouge_result_path', default='', required=True,
                        help='training or testing of MatchSum', type=str)
    args = parser.parse_known_args()[0]
    process(args)
