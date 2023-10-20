import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence
import string

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers
from tqdm import tqdm
import trlx.utils.logging as logging

from beautiful_prompt.utils import read_json, is_rank_0

logger = logging.get_logger()


IGNORE_INDEX = -100

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int = 512):
        super(SFTDataset, self).__init__()
        logger.info("Loading data...")
        
        data = read_json(data_path)

        new_data = []
        for d in data:
            if d['pick_score'] < 18.5:
                continue

            token_len = len(tokenizer.tokenize(d['prompt']))
            if token_len < 25:
                continue
            if token_len < 35 and random.random() < 0.3:
                continue
            new_data.append(d)
        
        data = new_data

        sources = [f"Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {d['raw_prompt']}\nOutput: " for d in data]
        
        targets = [d['prompt'].strip() + tokenizer.eos_token for d in data]
        
        logger.info(f'Num examples: {len(data)}')
        logger.info(f'Example 1: {sources[0]}{targets[0]}')

        logger.info("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class RMDatasetForAES(Dataset):
    """
    Dataset for reward model for aesthetic score

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        super().__init__()

        logger.info("Loading data...")
        
        data = read_json(data_path)

        self.inputs = []

        for d in tqdm(data, disable=not is_rank_0()):
            inp = d['prompt'] + tokenizer.eos_token
            inp = tokenizer(inp,
                            max_length=max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt")
            self.inputs.append({
                "input_ids": inp['input_ids'][0],
                "labels": torch.tensor(d['aesthetic_score'])
            })

    def __len__(self):
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.inputs[idx]["input_ids"], labels=self.inputs[idx]["labels"])

class RMDatasetForPS(Dataset):
    """
    Dataset for reward model for Pick Score

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> None:
        super().__init__()

        logger.info("Loading data...")
        
        data = read_json(data_path)

        self.inputs = []

        for d in tqdm(data, disable=not is_rank_0()):
            inp = f"Input: {d['raw_prompt']}\nOutput: {d['prompt']}{tokenizer.eos_token}"
            inp = tokenizer(inp,
                            max_length=max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt")
            self.inputs.append({
                "input_ids": inp['input_ids'][0],
                "labels": torch.tensor(d['pick_score']),
                # "labels": torch.tensor(d['image_reward'])
            })

    def __len__(self):
        length = len(self.inputs)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.inputs[idx]["input_ids"], labels=self.inputs[idx]["labels"])

@dataclass
class DataCollatorForRMDataset(DataCollatorForSFTDataset):
    """Collate examples for reward model."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

