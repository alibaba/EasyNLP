import argparse
from easynlp.appzoo import GeneralDataset, load_dataset

def parse_args():
    '''Parsing input arguments.'''
    parser = argparse.ArgumentParser(description="Arguments for load data.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/root/.easynlp/modelzoo/public/bert-base-uncased",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="clue/afqmc",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path
    task = args.datasets.split("/")[0]
    sub_set = args.datasets.split("/")[1]
    dataset = load_dataset(task, sub_set)["train"]
    encoded_dataset = GeneralDataset(dataset, tokenizer_path, args.seq_length)
    print(encoded_dataset[0])

if __name__ == "__main__":
    main()

