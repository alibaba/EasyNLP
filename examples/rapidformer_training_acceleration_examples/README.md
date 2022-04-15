<!---
Copyright 2021 The PAI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🤗 Rapidformer was created for PyTorch users who want to simply use accleration tricks powerd by DeepSpeed and Megatron.

<p align="center">
    <br>
    <img src="https://img.alicdn.com/imgextra/i3/O1CN01P3JMTs1dZXzGmKisb_!!6000000003750-2-tps-1361-1201.png" width="800" height="500"/>
    <br>
<p>

## Easy to integrate with No Trainer code, Below is an example:

```diff
  import torch
  import torch.utils.data import DataLoader
  from datasets import load_dataset, load_metric
  from transformers import (
  		AutoModelForSequenceClassification, 
      AutoTokenizer, 
      set_seed
  )
  
+ from rapidformer import RapidformerEngine
+ engine = RapidformerEngine()

  tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  datasets = load_dataset("glue", "mrpc")
  metric = load_metric("glue", "mrpc")
  
  def tokenize_function(examples):
     outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
     return outputs 
  tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
  tokenized_datasets.rename_column_("label", "labels")
  
  def collate_fn(examples):
  	return tokenizer.pad(examples, padding="longest", return_tensors="pt")
  	
    train_dataloader = DataLoader(
  	tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=micro_batch_size
  )
  eval_dataloader = DataLoader(
  	tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=micro_batch_size
  )
  
  optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

- lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )
    
+ lr_scheduler = partial(
        get_linear_schedule_with_warmup,
        num_warmup_steps=args.lr_warmup_iters,
        num_training_steps=args.train_iters
    )

+ model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = engine.compose(
        model=model, train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader, optimizer=optimizer, lr_scheduler_fn=lr_scheduler)
    
  for epoch in range(10):
  		model.train()
      for step, batch in enumerate(train_dataloader):
      		outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         engine.backward(loss)
          lr_scheduler.step()
          optimizer.step()
          optimizer.zero_grad()
```

As you can see in this example, by adding 5-lines to any standard PyTorch training script you can now run on any kind of single or distributed node setting as well as with mixed precision (fp16), zero optimizer or other accelerate tricks.

## Leveraging rapidformer preTrainer & finetuner to accelerate your training process.

### Accererating pretraining for easynlp models, below is code template, you can find more details in [rf_pretrain_easynlp_bert.py](rf_pretrain_easynlp_bert.py).

```python
from rapidformer import RapidformerEngine, PreTrainer

class EasyNLPRoBertaPreTrainer(PreTrainer):
    
    def train_valid_test_datasets_provider(self):
        pass
    
    def model_optimizer_lr_scheduler_provider(self):
        pass
    
    def run_forward_step(self, batch, model):
        pass
    
if __name__ == "__main__":
    engine = RapidformerEngine()
    trainer = EasyNLPRoBertaPreTrainer(engine=engine)
    trainer.train()

```

### Editing speedup arguments such as mixed_precision training or zero optimizer in script [run_pretrain_easynlp_bert.sh](run_pretrain_easynlp_bert.sh). you can fine more rapidformer configuration details in [Rapidformer Documents](https://help.aliyun.com/document_detail/406377.html).

### After done that, just run your script on your GPU devices for distributed pretraining.
```bash
bash run_pretrain_easynlp_bert.sh
```

### Accererating finetuning is as simple as pretraining, below is the code template.  
```python
from rapidformer import RapidformerEngine, Finetuner

class EasyNLPFintuner(Finetuner):
    
    def train_valid_test_datasets_provider(self):
        pass
    
    def model_optimizer_lr_scheduler_provider(self):
        pass
    
    def run_forward_step(self, batch, model):
        pass
    
    def run_compute_metrics(self, model, eval_dataloader):
        pass

if __name__ == "__main__":
    engine = RapidformerEngine()
    trainer = EasyNLPFintuner(engine=engine)
    trainer.train()

```
