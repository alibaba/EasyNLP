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

## Easy to integrate with No Trainer code
Here is an example:

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


## Simple and easily customisable Trainer for acceleration
```python
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from rapidformer import HuggingFaceTrainer, RapidformerEngine

    engine = RapidformerEngine()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    metric = load_metric("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
    tokenized_datasets.rename_column_("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    trainer = HuggingFaceTrainer(
        engine=engine,
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=metric,
        data_collator=collate_fn
    )

    trainer.train()
        

```

## Examples

The [run_rapidformer_no_trainer.sh](examples/nlp/run_rapidformer_no_trainer.sh) script is a simple example to train a Bert model on a classification task ([GLUE's MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)).

The [run_rapidformer_no_trainer.sh](examples/cv/run_rapidformer_no_trainer.sh) script is a simple example to fine-tune a ResNet-50 on a classification task ([Ofxord-IIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)).

Prior to running it you should install timm and torchvision:
```bash
pip install timm torchvision
```

and you should download the data with the following commands:

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xzf images.tar.gz
```