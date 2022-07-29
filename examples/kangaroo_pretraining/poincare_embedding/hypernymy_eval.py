#!/usr/bin/env python3

import json
from hype.hypernymy_eval import main as hype_eval
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    opt = parser.parse_args()

    results, summary = hype_eval(opt.file, cpu=False)

    print(json.dumps(results))
