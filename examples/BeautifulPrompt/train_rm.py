import argparse
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from beautiful_prompt.data import RMDatasetForAES, RMDatasetForPS, DataCollatorForRMDataset
from beautiful_prompt.utils import set_seed
from beautiful_prompt.trainer import RMTrainer

def train(args):
    set_seed(args.seed)
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    tokenizer.truncation_side = 'left'
    tokenizer.padding_side = 'right'

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.rm_type.lower() == 'aes':
        train_dataset = RMDatasetForAES(args.data_path, tokenizer, max_length=args.max_length)
    elif args.rm_type.lower() == 'ps':
        train_dataset = RMDatasetForPS(args.data_path, tokenizer, max_length=args.max_length)
    else:
        raise NotImplementedError()

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=True,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
    else:
        train_sampler = None
    
    data_collator = DataCollatorForRMDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator,
                                  pin_memory=True)
    
    trainer = RMTrainer(
        model,
        tokenizer,
        train_dataloader,
        save_path=args.save_path,
        logging_dir=args.logging_dir,

        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str, default='bigscience/bloom-1b1')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--max_length', type=int, default=384)
    
    parser.add_argument('--save_path', type=str, default='outputs/rm')
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    
    parser.add_argument('--rm_type', type=str, default='aes')
    
    args = parser.parse_args()
    train(args)
