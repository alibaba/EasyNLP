# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 2:56 pm.
# @Author  : JianingWang
# @File    : mlm_pretokenize.py
from itertools import chain
from datasets import load_dataset
from transformers import AutoTokenizer

train_file = '/home/wenkang.hwk/nas/datasets/datasets/mlm/baike_book_emr_maintell400w/train.txt'
validation_file = '/home/wenkang.hwk/nas/datasets/datasets/mlm/baike_book_emr_maintell400w/validation.txt'

validation_split_percentage = 2
max_seq_length = 512
preprocessing_num_workers = 20
cache_dir = None
do_train = True
tokenizer = AutoTokenizer.from_pretrained('/home/wenkang.hwk/nas/models/pretrain_task/med_roformer_base_add_vocab')

data_files = {}
if train_file is not None:
    data_files["train"] = train_file
if validation_file is not None:
    data_files["validation"] = validation_file
extension = "text"
raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)
# raw_datasets['train'] = raw_datasets['train'].shuffle()
# If no validation data is there, validation_split_percentage will be used to divide the dataset.
if "validation" not in raw_datasets.keys():
    raw_datasets["validation"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train[:{validation_split_percentage}%]",
        cache_dir=cache_dir,
    )
    raw_datasets["train"] = load_dataset(
        extension,
        data_files=data_files,
        split=f"train[{validation_split_percentage}%:]",
        cache_dir=cache_dir,
    )

# Preprocessing the datasets.
# First we tokenize all the texts.
if do_train:
    column_names = raw_datasets["train"].column_names
else:
    column_names = raw_datasets["validation"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]
max_seq_length = tokenizer.model_max_length if max_seq_length is None else max_seq_length


def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)


print('tokenize')
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on every text in dataset",
)


# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    result = {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


raw_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=preprocessing_num_workers,
    load_from_cache_file=False,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)

raw_datasets.save_to_disk('/home/wenkang.hwk/nas/datasets/mlm/baike_book_emr_maintell400w')
