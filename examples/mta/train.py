# -*- coding: utf-8 -*-
import os,json
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer , AutoConfig
from transformers.generation import utils
from rich.table import Column, Table
from rich import box
from rich.console import Console
import argparse
import tqdm
import json,pylcs
import prettytable as pt
from modeling_MTA import T5ForConditionalGenerationMTA
from script.eval import evaluate_pclue_fn_no_mrc
from script.forward_val import forward_predict
from predict import predict

import logging

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank",type=int,default=0)
parser.add_argument("--csv_path",type=str,default="project/data/clean_data_ori/english_train_new.csv",help="训练数据转为csv")
parser.add_argument("--model_save_path",type=str,default="./result/",help="模型存储的位置")
parser.add_argument("--model_name",type=str,default="for_debug",help="模型存储的文件夹名")
parser.add_argument("--model_for_train",type=str,default='t5-base')
parser.add_argument("--batch_size",type=int,default=4)
parser.add_argument("--train_epoch",type=int,default=1)
parser.add_argument("--val_epoch",type=int,default=1)
parser.add_argument("--accumulate",type=int,default=None)
parser.add_argument("--val_json_path",type=str,help="验证集地址",default="project/data/clean_data_ori/english_val_new_shuffle.json")
parser.add_argument("--config_path",type=str,help="验证集地址",default="models/test_config/t5_base_config.json")
parser.add_argument("--eval_json_path",type=str,default="project/data/clean_data_ori/english_val_new_shuffle.json")
parser.add_argument("--source_max_len",type=int,default=512)
parser.add_argument("--random_seed",type=int,default=42)
parser.add_argument("--lr",type=float,default=1e-4)
args = parser.parse_args()

# model parameters
model_for_train =args.model_for_train
lr = args.lr
source_max_len = args.source_max_len
train_epoch = args.train_epoch
batch_size = args.batch_size
val_epoch = args.val_epoch
eval_json_path = args.eval_json_path
random_seed = args.random_seed
config_path = args.config_path
val_json_path = args.val_json_path

accumulate = args.accumulate
csv_path = args.csv_path
model_name = args.model_name
model_save_path = args.model_save_path

# 定义模型的参数 let's define model parameters specific to T5
model_params = {
    "MODEL": model_for_train,  # model_type
    "TRAIN_BATCH_SIZE": batch_size,  # training batch size, 8
    "TRAIN_EPOCHS": train_epoch,  # number of training epochs
    "VAL_EPOCHS": val_epoch,  # number of validation epochs
    "LEARNING_RATE": lr,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": source_max_len,  # max length of source text, 512
    "MAX_TARGET_TEXT_LENGTH": 64,  # max length of target text,64
    "SEED": random_seed,  # set seed for reproducibility
}

# define a rich console logger
console = Console(record=True)

# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

class YourDataSetClass(Dataset):
    """
    创建一个自定义的数据集，用于训练，必须包括两个字段：输入(如source_text)、输出（如target_text）
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]
        self.type_text = self.data["type"]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        type_label = str(self.type_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())
        type_label = " ".join(type_label.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len-1,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "type_ids":  type_label,
        }

def train(epoch, tokenizer, model, device, loader, optimizer,path_final,val_json_path,new_token):

    """
    用于训练的方法
    Function to be called for training with the parameters passed from main function

    """
    print("start train")
    eval_step = 7000
    score_temp = 0
    select_top = -1
    for i, data in enumerate(tqdm.tqdm(loader), 0):
        model.train()

        data["target_ids"] = torch.cat((torch.tensor([[new_token]]*data["target_ids"].shape[0]),data["target_ids"]),1)
        
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous() # target, from start to end(except end of token, ). e.g. "你好吗？"
        lm_labels = y[:, 1:].clone().detach() # target, for second to end.e.g."好吗？"
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100 # releted to pad_token and loss. for detail, check here: https://github.com/Shivanandroy/T5-Finetuning-PyTorch/issues/3
        ids = data["source_ids"].to(device, dtype=torch.long) # input. e.g. "how are you?"
        mask = data["attention_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
            type_label=data["type_ids"],
        )
        loss = outputs[0]

        if (i+1)%300 == 0:
            print(str(epoch), str(i), str(loss))

        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

        if (i) % eval_step == 0 and (epoch % 2 == 1):
            model.eval()
            with torch.no_grad():
                file_path = os.path.join(path_final,"{step}_predictions.json".format(step=i+1))
                forward_predict(source_file=val_json_path,target_file=file_path,tokenizer=tokenizer,model=model,select_top=select_top)
                score = evaluate_pclue_fn_no_mrc(file_path,val_json_path,select_top=select_top)
                if score > score_temp:
                    model.save_pretrained(os.path.join(path_final,"best_model"))
                    tokenizer.save_pretrained(os.path.join(path_final,"best_model"))
                    score_temp = score
    print("select_top:",select_top)
# 训练类：整合数据集类、训练方法、验证方法，加载数据进行训练并验证训练过程的效果
def T5Trainer(
    local_rank,dataframe,source_text, target_text, model_params,val_json_path,eval_json_path
):
    """
    T5 trainer
    """
    device = local_rank

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy randdsdom seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    config = AutoConfig.from_pretrained(config_path)
    config.foo = 'bar'

    model = T5ForConditionalGenerationMTA.from_pretrained(model_params["MODEL"],config=config)
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    model.Copyweight()

    model = model.to(device)

    tokenizer.add_special_tokens({'additional_special_tokens':["<e>"]})
    new_token = tokenizer.encode(["<e>"])[0]

    path_final = model_save_path + model_name
    print(torch.cuda.current_device())
    if torch.cuda.current_device() == 0:
        print(path_final)
        assert not os.path.exists(path_final) , "模型存储路径已存在需要修改"
        os.mkdir(path_final)

    # logging
    console.log(f"[Data]: Reading data...\n")

    train_dataset = dataframe

    # 打印数据集相关日志：数据量、训练步数
    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    total_train_steps=int((train_dataset.shape[0] * model_params["TRAIN_EPOCHS"])/model_params["TRAIN_BATCH_SIZE"])
    console.print(f"Total Train Steps: {total_train_steps}\n")
    
    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, **train_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader,optimizer ,path_final ,val_json_path,new_token)
        console.log(f"[Saving Model]...\n")
        path = os.path.join(path_final, "final")
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    with torch.no_grad(): 
        eval_best_model = os.path.join(path_final,"best_model")
        predict(eval_best_model,eval_json_path)

df = pd.read_csv(csv_path)  # 数据量：1200k数据。

T5Trainer(
    local_rank=args.local_rank,
    dataframe=df,
    source_text="input",
    target_text="target",
    val_json_path=val_json_path,
    eval_json_path=eval_json_path,
    model_params=model_params,
)
print("train finished end..")
