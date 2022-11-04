# -*- coding: utf-8 -*-
# @Time    : 2022/1/7 3:07 pm.
# @Author  : JianingWang
# @File    : HFTrainer
from typing import Dict, Union, Any, Optional, Callable, List, Tuple, Iterator

import datasets
from datasets import Dataset
from torch.utils.data import RandomSampler, DistributedSampler
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.trainer_pt_utils import DistributedSamplerWithLoop, get_length_grouped_indices
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler as DistributedLengthGroupedSamplerOri
from transformers.trainer_pt_utils import LengthGroupedSampler as LengthGroupedSamplerOri
# from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode

from config import TrainingArguments
from transformers.trainer import Trainer, _is_torch_generator_available
import torch
from torch import nn
from transformers.file_utils import is_datasets_available
from models.adversarial import FGM


class LengthGroupedSampler(LengthGroupedSamplerOri):
    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator, mega_batch_mult=256)
        return iter(indices)


class DistributedLengthGroupedSampler(DistributedLengthGroupedSamplerOri):
    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g, mega_batch_mult=400)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class HFTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super(HFTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)
        if self.args.do_adv:
            self.fgm = FGM(self.model)
        for callback in callbacks:
            callback.trainer = self

        self.global_step_ = 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        self.global_step_ += 1
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.global_step_ % 10 == 0:
            print('[step={}, loss={}]'.format(self.global_step_, loss))

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()

        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        # adv
        if self.args.do_adv:
            self.fgm.attack()
            with self.autocast_smart_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            if self.do_grad_scaling:
                self.scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            self.fgm.restore()

        return loss.detach()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # if not has_length(self.train_dataset):
        #     return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset, generator=generator)
                return RandomSampler(self.train_dataset)
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
