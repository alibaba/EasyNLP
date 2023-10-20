import os
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import logging
import math

from sklearn.metrics import f1_score, average_precision_score
import numpy as np


logger = logging.getLogger(__name__)

def save_model(args, model, tokenizer):
    # Create output directory if needed

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


class BatchContinuousRandomSampler(RandomSampler):
    """ make sure examples with same language in batch """

    def __init__(self, data_source, replacement: bool = False,
                 num_samples = None, generator=None, batch_size=None) -> None:
    
        super().__init__(data_source, replacement, num_samples, generator)
        self.batch_size = batch_size

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            l = torch.randperm(n // self.batch_size, generator=self.generator).tolist()
            l2 = []
            for x in l:
                for i in range(self.batch_size):
                    l2.append(x * self.batch_size + i)

            yield from l2

class BatchContinuousDistributedSampler(DistributedSampler):
    """ make sure examples with same language in batch """

    def __init__(self, dataset, num_replicas = None,
                 rank = None, shuffle = True,
                 seed = 0, drop_last = False, batch_size=None) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size

        n = len(dataset)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(n // self.batch_size, generator=g).tolist()  # type: ignore
        new_indices = []
        for x in indices:
            for i in range(self.batch_size):
                new_indices.append(x * self.batch_size + i)
        
        self.indices = new_indices[self.rank:self.total_size:self.num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class AttentionTeacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        attention_mask=None,

        query=None, # target_hidden_states
        key=None, # source_hidden_states
        value=None, # source_logits
    ):
        
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # batch_size, num_head, seq_len, seq_len

        output = torch.matmul(attention_probs, value.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1))
        output = torch.mean(output, dim=1, keepdim=True).squeeze(1)

        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, config, temp=0.05):
        super().__init__()
        self.mlp = MLPLayer(config)
        self.temp = temp

        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        x = self.mlp(x)
        y = self.mlp(y)

        cos_sim = self.cos(x.unsqueeze(1), y.unsqueeze(0)) / self.temp

        labels = torch.arange(cos_sim.size(0)).long().to(x.device)
        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(cos_sim, labels)
        return loss

class BatchNorm(nn.Module):
    def __init__(self, hidden_size=768, eps=1e-8, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.hidden_size = hidden_size
        
        self.register_buffer('running_mean', torch.zeros(hidden_size))
        self.register_buffer('running_var', torch.ones(hidden_size))

    def forward(self, input, attention_mask=None):
        if self.training:
            exponential_average_factor = self.momentum

            mean = input.mean((0, 1))
            var = input.var((0, 1), unbiased=False)

            if attention_mask is not None:
                mean = ((input * attention_mask[:, :, None]).sum(1) / attention_mask.sum(-1)[:, None]).mean(0)
                var = torch.pow(input * attention_mask[:, :, None] \
                                - mean[None, None, :] * attention_mask[:, :, None], 2).sum((0, 1)) \
                                / attention_mask.sum()
            else:
                mean = input.mean((0, 1))
                var = input.var((0, 1), unbiased=False)

            with torch.no_grad():                        
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * var \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        return (input - mean[None, None, :]) / torch.sqrt(var[None, None, :] + self.eps)

def get_attention_entropy(attention_score):
    """Get attention entropy based on attention score."""

    bz, n_heads, seq_len_q, seq_len_k = attention_score.size()
    attention_score = attention_score.mean(dim=1)
    # (batch size, seq_len_q, seq_len_k)
    attention_entropy = -(attention_score * torch.log(attention_score + 1e-8))
    attention_entropy = attention_entropy.sum(dim=-1)

    mean_attention_entropy = attention_entropy.mean(dim=-1)

    return mean_attention_entropy

def get_pair_entropy(attention_score):
    """Get pairwise attention entropy."""
    entropy1 = get_attention_entropy(attention_score)
    attention_score = attention_score.permute(0, 1, 3, 2)
    entropy2 = get_attention_entropy(attention_score)

    return entropy1 + entropy2
