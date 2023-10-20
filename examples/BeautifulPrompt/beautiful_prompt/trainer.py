import math
import time
from abc import ABC

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import get_scheduler
import trlx.utils.logging as logging
from accelerate import Accelerator

from beautiful_prompt.utils import get_optimizer_grouped_parameters

logger = logging.get_logger()

class SFTTrainer(ABC):
    """
    SFTTrainer

    Args:
        model (torch.nn.Module):The model to be trained.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to preprocess the input data.
        train_dataloader (DataLoader): The dataloader containing the training data.
        save_path (str): The path to save the trained model.
        logging_dir (str): The directory to save the training logs. Defaults to 'logs'.
        lr (float): The learning rate for the optimizer. Defaults to 1e-5.
        batch_size (int): The batch size for training. Defaults to 8.
        weight_decay (float): The weight decay for the optimizer. Defaults to 1e-3.
        epochs (int): The number of epochs for training. Defaults to 3.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataloader: DataLoader,
        save_path: str,
        logging_dir: str = 'logs',
        
        lr: float = 1e-5,
        batch_size: int = 8,
        weight_decay: float = 1e-3,
        epochs: int = 3
    ) -> None:
        super().__init__()
        
        self.accelerator = Accelerator(log_with="tensorboard", project_dir=logging_dir)
        ds_plugin = self.accelerator.state.deepspeed_plugin

        self.model = model
        self.tokenizer = tokenizer
        self.save_path = save_path

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95))
        
        self.train_dataloader = train_dataloader

        self.epochs = epochs

        self.accumulation_steps = ds_plugin.gradient_accumulation_steps
        
        num_update_steps_per_epoch = len(train_dataloader) // self.accumulation_steps
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)

        self.scheduler = get_scheduler("cosine",
                                       self.optimizer,
                                       num_warmup_steps=math.ceil(max_steps * 0.03),
                                       num_training_steps=max_steps)

        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.scheduler
        )
        
        self._init_logger(lr, batch_size, epochs, ds_plugin)
        
    def _init_logger(self, lr, batch_size, epochs, ds_plugin):
        if self.accelerator.is_main_process:
            config = {
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "mixed_precision": self.accelerator.mixed_precision,
                "num_gpus": self.accelerator.num_processes,
                "gradient_accumulation_steps": ds_plugin.gradient_accumulation_steps,
                "gradient_clipping": ds_plugin.gradient_clipping,
                "zero_stage": ds_plugin.zero_stage,
            }

            self.accelerator.init_trackers(
                project_name=f'beautilful-prompt sft [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]',
                config=config
            )

    def train(self):
        step_bar = tqdm(range(len(self.train_dataloader) // self.accumulation_steps * self.epochs),
                        desc=f'steps',
                        disable=not self.accelerator.is_main_process)

        current_step = 0
        for epoch in range(self.epochs):

            self.model.train()
            for batch_id, batch in enumerate(self.train_dataloader):
                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)

                input_ids = batch["input_ids"].to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = batch["labels"].to(torch.cuda.current_device())
                
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.accelerator.log({
                            "loss": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "batch_id": batch_id
                        }, step=current_step)

                        current_step += 1
                        step_bar.update()

        step_bar.close()
        self.save_model(self.save_path)

    def save_model(self, save_path: str) -> None:
        self.accelerator.wait_for_everyone()
        self.accelerator.unwrap_model(self.model).save_pretrained(
            save_path,
            save_function=self.accelerator.save,
            is_main_process=self.accelerator.is_main_process,
            state_dict=self.accelerator.get_state_dict(self.model)
        )

        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(save_path)

class RMTrainer(SFTTrainer):
    """
    Trainer to use while training reward model.

    Args:
        model (torch.nn.Module):The model to be trained.
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to preprocess the input data.
        train_dataloader (DataLoader): The dataloader containing the training data.
        eval_dataloader (DataLoader): The dataloader containing the evaluate data.
        save_path (str): The path to save the trained model.
        logging_dir (str): The directory to save the training logs. Defaults to 'logs'.
        eval_steps (int): The interval to evaluate the model during training. Defaults to 1000.
        lr (float): The learning rate for the optimizer. Defaults to 1e-5.
        batch_size (int): The batch size for training. Defaults to 8.
        weight_decay (float): The weight decay for the optimizer. Defaults to 1e-3.
        epochs (int): The number of epochs for training. Defaults to 3.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataloader: DataLoader,
        save_path: str,
        eval_dataloader: DataLoader = None,
        logging_dir: str = 'logs',
        eval_steps: int = 1000,

        lr: float = 1e-5,
        batch_size: int = 8,
        weight_decay: float = 1e-3,
        epochs: int = 3
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            train_dataloader,
            save_path,
            logging_dir,
            lr,
            batch_size,
            weight_decay,
            epochs
        )
        
        if eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(eval_dataloader)
            self.eval_steps = eval_steps
        else:
            self.eval_steps = -1

    def _init_logger(self, lr, batch_size, epochs, ds_plugin):
        if self.accelerator.is_main_process:
            config = {
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "mixed_precision": self.accelerator.mixed_precision,
                "num_gpus": self.accelerator.num_processes,
                "gradient_accumulation_steps": ds_plugin.gradient_accumulation_steps,
                "gradient_clipping": ds_plugin.gradient_clipping,
                "zero_stage": ds_plugin.zero_stage,
            }

            self.accelerator.init_trackers(
                project_name=f'beautilful-prompt rm [{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]',
                config=config
            )

    def train(self):
        step_bar = tqdm(range(len(self.train_dataloader) // self.accumulation_steps * self.epochs),
                        desc=f'steps',
                        disable=not self.accelerator.is_main_process)

        current_step = 0
        for epoch in range(self.epochs):

            self.model.train()
            for batch_id, batch in enumerate(self.train_dataloader):
                if isinstance(self.train_dataloader.sampler, DistributedSampler):
                    self.train_dataloader.sampler.set_epoch(epoch)

                input_ids = batch["input_ids"].to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = torch.tensor(batch["labels"], dtype=self.model.dtype).to(torch.cuda.current_device())
                
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        self.accelerator.log({
                            "loss": loss.item(),
                            "lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "batch_id": batch_id
                        }, step=current_step)

                        current_step += 1
                        step_bar.update()
                
                if self.eval_steps > 0 and (current_step + 1) % self.eval_steps:
                    result = self.eval()
                    self.accelerator.log(result, step=current_step)
        
        step_bar.close()
        self.save_model(self.save_path)

    def eval(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch["input_ids"].to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = torch.tensor(batch["labels"], dtype=self.model.dtype).to(torch.cuda.current_device())
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
                loss = outputs.loss
                total_loss += loss
            total_loss = total_loss / len(self.eval_dataloader)
            
        self.model.train()
        return {
            "mse": total_loss,
        }
