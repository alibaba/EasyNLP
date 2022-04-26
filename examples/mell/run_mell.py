# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from collections import OrderedDict, defaultdict
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import layers, losses
from tokenization import Tokenizer
from trainer import Trainer
from optimizers import BertAdam
from basedataset import BaseDataset
from bert_preprocessors import InputExample, bert_cls_convert_example_to_feature
from utils import logger, config, io, get_dir_name, init_running_envs, distributed_call_main


class MetaIntentDataset(BaseDataset):
    def __init__(self,
                 data_file,
                 meta_info_file,
                 vocab_file,
                 max_seq_length,
                 max_label_num=10,
                 **kwargs):
        super(MetaIntentDataset, self).__init__(data_file, **kwargs)
        self.tokenizer = Tokenizer(backend="bert", vocab_file=vocab_file)
        self.max_seq_length = max_seq_length
        self.max_label_num = max_label_num

        with io.open(meta_info_file) as f:
            meta_info_json = eval(json.load(f))['data']

        self.task_to_idx = dict()
        self.task_to_label_mapping = dict()
        self.task_to_label_features = dict()
        self.label_to_memory_id = {"PAD": 0}

        for task_label_info in meta_info_json:
            labels = task_label_info["labelMap"]

            # 任务中包含的标签
            label_map = {label: idx for idx, label in enumerate(labels)}

            # task_key: 任务名
            task_key = task_label_info["taskKey"]

            self.task_to_idx[task_key] = len(self.task_to_idx)
            self.task_to_label_mapping[task_key] = label_map

            for label in labels:
                # 注意这里有可能出现不同的任务对应的label是一样的名字，但是只要是在同一个dataset下面，就是默认同一个label 名字就是一个意思的表达
                if label not in self.label_to_memory_id:
                    self.label_to_memory_id[label] = len(self.label_to_memory_id)

    @property
    def eval_metrics(self):
        return ("accuracy", "macro-f1", "micro-f1")

    def convert_single_row_to_example(self, row):
        items = row.strip().split("\t")
        task, content, label = items[0], items[1], items[2]
        example = InputExample(text_a=content, text_b=None, label=label)
        feature = bert_cls_convert_example_to_feature(example, self.tokenizer, self.max_seq_length, self.task_to_label_mapping[task])
        setattr(feature, "task", task)
        return feature

    def batch_fn(self, features):
        label_memory_ids = list()
        label_memory_masks = list()
        for f in features:
            tmp = [self.label_to_memory_id[label] for label in self.task_to_label_mapping[f.task]]
            tmp = tmp[:self.max_label_num]
            tmp_label_memory_mask = [1] * len(tmp)
            tmp = tmp + [0] * (self.max_label_num - len(tmp))
            tmp_label_memory_mask += [0] * (self.max_label_num - len(tmp_label_memory_mask))
            label_memory_masks.append(tmp_label_memory_mask)
            label_memory_ids.append(tmp)

        inputs = {
            "input_ids": torch.tensor([f.input_ids for f in features], dtype=torch.long),
            "input_mask": torch.tensor([f.input_mask for f in features], dtype=torch.long),
            "segment_ids": torch.tensor([f.segment_ids for f in features], dtype=torch.long),
            "label_ids": torch.tensor([f.label_id for f in features], dtype=torch.long),
            "task_ids": [self.task_to_idx[f.task] for f in features],
            "tasks": [f.task for f in features],
            "label_memory_ids": label_memory_ids,
            "label_memory_mask": torch.tensor(label_memory_masks, dtype=torch.long)
        }
        return inputs


class MetaLabelEnhancedBertClassify(layers.BaseModel):
    def __init__(self, config, **kwargs):
        super(MetaLabelEnhancedBertClassify, self).__init__(config, **kwargs)
        self.model_name = "text_classify_meta_label_enhanced_bert"

        if config.model_type == "bert":
            self.bert = layers.BertModel(config)
        elif config.model_type == "albert":
            logger.info("Use ALBERT model")
            self.albert = layers.AlbertModel(config)
        else:
            raise NotImplementedError

        self.max_memory_size = kwargs["max_memory_size"]
        self.max_task_num = kwargs["max_task_num"]
        self.max_label_num = kwargs["max_label_num"]
        self.freeze_encoder = kwargs["freeze_encoder"]
        self.is_testing = kwargs.get("is_testing", False)

        # global memory
        self.memory = nn.Parameter(torch.zeros((self.max_memory_size, config.hidden_size)))

        # three-dim= task * task_label * hidden_size
        self.local_memories = nn.Parameter(torch.zeros((self.max_task_num, self.max_label_num, config.hidden_size)))
        self.local_memories.requires_grad = False

        # max_memory_size: 41, max_task_num: 100, max_label_num: 10
        self.memory_id_to_task_count = {idx: 0 for idx in range(self.max_memory_size)}

        # 重复的语句？
        self.local_memories = nn.Parameter(torch.zeros((self.max_task_num, self.max_label_num, config.hidden_size)))

        self.output = nn.ParameterDict({
            "multi_task_kernel": nn.Parameter(torch.randn((self.max_task_num, self.max_label_num, config.hidden_size)).normal_(std=0.02)),
            "multi_task_bias": nn.Parameter(torch.zeros((self.max_task_num, self.max_label_num)))
        })

        # freeze_encoder = False
        if self.freeze_encoder:
            if config.model_type == "bert":
                for name, param in self.bert.named_parameters():
                    param.requires_grad = False
            elif config.model_type == "albert":
                for name, param in self.albert.named_parameters():
                    param.requires_grad = False


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_model_weights)

    def fetch_label_memory_embedding(self, label_memory_ids, task_ids):
        batch_size = len(label_memory_ids)
        max_label_size = len(label_memory_ids[0])
        if not self.is_testing:
            flattened_label_memory_ids = [t for item in label_memory_ids for t in item]
            label_memory_embedding = self.memory[flattened_label_memory_ids].view(
                batch_size, max_label_size, -1)
        else:
            label_memory_embedding = self.local_memories[task_ids]
        return label_memory_embedding

    def forward(self, inputs):
        if self.config.model_type == "bert":
            sequence_output, _, pooled_output = \
                self.bert(inputs["input_ids"],
                          inputs["segment_ids"],
                          inputs["input_mask"],
                          output_all_encoded_layers=True, output_att=True)
        elif self.config.model_type == "albert":
            outputs = self.albert(input_ids=inputs["input_ids"],
                                  token_type_ids=inputs["segment_ids"],
                                  attention_mask=inputs["input_mask"])
            pooled_output = outputs.pooler_output
            pooled_output = self.dropout(pooled_output)
        else:
            raise NotImplementedError

        # MetaLabelEnhancedBertClassify + LRUMetaLabelEnhancedBertClassify（继承这个forward函数）
        # 这里因为加载了embedding，所以会直接更新global memory
        label_memory_embedding = self.fetch_label_memory_embedding(
            inputs["label_memory_ids"], inputs["task_ids"])

        dots = torch.bmm(label_memory_embedding, pooled_output.unsqueeze(2))
        dots.masked_fill_((1 - inputs["label_memory_mask"].unsqueeze(-1)).bool(), -1e8)
        weights = torch.softmax(dots, dim=1)
        label_embedding = torch.bmm(label_memory_embedding.transpose(1, 2), weights)

        # [batch_size, max_label_num, hidden_size]
        batch_kernel = self.output.multi_task_kernel[inputs["task_ids"], :, :]
        # [batch_size, max_label_num]
        batch_bias = self.output.multi_task_bias[inputs["task_ids"], :]

        final_output = pooled_output.unsqueeze(1) + label_embedding.transpose(1, 2)
        logits = torch.bmm(final_output, batch_kernel.transpose(1, 2))
        logits = logits.squeeze(1) + batch_bias

        return {
            "logits": logits
        }

    def compute_loss(self, model_outputs, inputs):
        logits = model_outputs["logits"]
        label_ids = inputs["label_ids"]
        return {
            "loss": losses.cross_entropy(logits, label_ids)
        }

    def update_global_memory(self, memory_id_to_label_embedding, _lambda=1.0):
        for memory_id, label_embedding in memory_id_to_label_embedding.items():
            if self.memory_id_to_task_count[memory_id] == 0:
                # 如果这个label对应的表示没有在memory池子里面，就直接添加进去（这个不是LRU）
                self.memory[memory_id].data.copy_(label_embedding)
            else:
                # 和paper不一致的地方(可能)
                # 这里更新是按照label ID对应的task数量进行的初始化，注意，后面除以了task 数量
                self.memory[memory_id].data.copy_(self.memory[memory_id].data * self.memory_id_to_task_count[memory_id]
                                                  + _lambda * label_embedding)
            self.memory_id_to_task_count[memory_id] += 1

            # 上面if else 已经对self.memory[memory_id].data这个embedding进行了更新，这里可能是一个scale的处理
            self.memory[memory_id].data.copy_(self.memory[memory_id].data / self.memory_id_to_task_count[memory_id])

    def update_local_memory(self, task_id, task_memory_label_ids):
        # task_id： dataset里面有很多个task，每个task对应一个ID
        # task_memory_ids：这个是task里面包含的所有label对应的ID
        task_global_memory = self.memory[task_memory_label_ids]
        task_label_num = len(task_memory_label_ids)

        # copy_()函数是数值覆盖和broadcast，梯度还是根据copy函数之前的原变量来确定
        # 将local对应的信息进行了更新，local进行inference的任务embedding已经是最新的结果
        self.local_memories[task_id, :task_label_num].data.copy_(task_global_memory.data)


class LRUMemory(object):
    def __init__(self, memory_slot):
        self.memory_slot = memory_slot
        self.max_memory_size = memory_slot.size(0)

        # 是一个有序的数据字典结构
        self.lru_cache = OrderedDict()

        self.key_to_count = defaultdict()

    def get_memories_without_updating(self, keys):
        rst = list()
        for key in keys:
            if key == 0:
                memory_slot_id = 0
            else:
                memory_slot_id = self.lru_cache[key]
            tmp_memory_slot = self.memory_slot[memory_slot_id].unsqueeze(0)
            rst.append(tmp_memory_slot)
        rst = torch.cat(rst, dim=0)
        return rst

    def set_memory(self, key, embedding):
        if key in self.lru_cache:
            memory_slot_id = self.lru_cache[key]
            old_memory = self.memory_slot[memory_slot_id]
            cnt = self.key_to_count[key]
            new_memory = (old_memory * cnt + embedding) / (cnt + 1)
            self.key_to_count[key] += 1
            self.lru_cache.pop(key)
            self.lru_cache[key] = memory_slot_id
            self.memory_slot[memory_slot_id].data.copy_(new_memory)
        else:
            if len(self.lru_cache) == self.max_memory_size:
                # pop 的是第一个元素，lru_cache是有序字典数据结构
                _, popped_memory_slot_id = self.lru_cache.popitem(last=False)
                self.lru_cache[key] = popped_memory_slot_id
                self.memory_slot[popped_memory_slot_id].data.copy_(embedding)
                self.key_to_count[key] = 1
            else:
                self.lru_cache[key] = len(self.lru_cache)
                self.memory_slot[self.lru_cache[key]].data.copy_(embedding)
                self.key_to_count[key] = 1

class LRUMetaLabelEnhancedBertClassify(MetaLabelEnhancedBertClassify):
    def __init__(self, config, **kwargs):
        super(LRUMetaLabelEnhancedBertClassify, self).__init__(config, **kwargs)
        self.model_name = "text_classify_lru_meta_label_enhanced_bert"
        self.memory_slot = nn.Parameter(torch.zeros((self.max_memory_size, config.hidden_size)))
        self.memory_ = LRUMemory(self.memory_slot)
        self.memory_.set_memory(0, torch.zeros(config.hidden_size))

    def fetch_label_memory_embedding(self, label_memory_ids, task_ids):
        batch_size = len(label_memory_ids)
        max_label_size = len(label_memory_ids[0])
        if not self.is_testing:
            flattened_label_memory_ids = [t for item in label_memory_ids for t in item]
            label_memory_embedding = self.memory_.get_memories_without_updating(
                flattened_label_memory_ids).view(batch_size, max_label_size, -1)
        else:
            label_memory_embedding = self.local_memories[task_ids]
        return label_memory_embedding

    def update_global_memory(self, memory_id_to_label_embedding, _lambda=1.0):
        for memory_id, label_embedding in memory_id_to_label_embedding.items():
            self.memory_.set_memory(memory_id, label_embedding)

    def update_local_memory(self, task_id, task_memory_ids):
        task_global_memory = self.memory_.get_memories_without_updating(task_memory_ids)
        task_label_num = len(task_memory_ids)
        self.local_memories[task_id, :task_label_num].data.copy_(task_global_memory.data)


def base_mtl_training(cfg, base_task_keys):
    if cfg.use_lru:
        print("use LRU model")
        # logger.info("use LRU model")
        model = LRUMetaLabelEnhancedBertClassify.from_pretrained(
            cfg.pretrain_model_name_or_path,
            max_memory_size=cfg.max_memory_size,
            max_task_num=cfg.max_task_num,
            max_label_num=cfg.max_label_num,
            freeze_encoder=False)
    else:
        model = MetaLabelEnhancedBertClassify.from_pretrained(
            cfg.pretrain_model_name_or_path,
            max_memory_size=cfg.max_memory_size,
            max_task_num=cfg.max_task_num,
            max_label_num=cfg.max_label_num,
            freeze_encoder=False)

    # 处理数据的时候，按照task为单位进行的处理和记录数据（meta-learning的基本单位），对应task中的sample representation是提前计算好的
    train_dataset = MetaIntentDataset(
        model_type="text_classify_bert",
        data_file=os.path.join(cfg.tables, "base_tasks.json" if cfg.base_k is None else "base_train_%d.tsv" % cfg.base_k),
        meta_info_file=os.path.join(cfg.tables, "meta_info.json"),
        vocab_file=get_dir_name(cfg.pretrain_model_name_or_path) + "/vocab.txt",
        max_seq_length=cfg.sequence_length,
        is_training=True)

    valid_dataset = MetaIntentDataset(
        model_type="text_classify_bert",
        data_file=os.path.join(cfg.tables, "base_tasks.json" if cfg.base_k is None else "base_dev_%d.tsv" % cfg.base_k),
        meta_info_file=os.path.join(cfg.tables, "meta_info.json"),
        vocab_file=get_dir_name(cfg.pretrain_model_name_or_path) + "/vocab.txt",
        max_seq_length=cfg.sequence_length,
        is_training=False)

    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir_base, "base")

    # 初始化整个模型运行过程
    trainer = Trainer(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, cfg=cfg)

    # 把当前需要进行training的task进行处理，因为是meta learner的training阶段，所以就是对global embedding进行初始化，然后利用training阶段进行更新
    for task_key in base_task_keys:
        # *****************************
        # 这里我们没有对应的label embedding, 所以先用随机初始化，应该按照论文中描述的，用模型来计算
        # with open(os.path.join(cfg.tables, task_key, "label_embeddings.json")) as f:
        #     label_embeddings = json.load(f)
        label_embeddings = load_label_emb(task_key)
        # *****************************
        memory_id_to_label_embedding = dict()
        for label, embedding in label_embeddings.items():
            memory_id = valid_dataset.label_to_memory_id[label]
            memory_id_to_label_embedding[memory_id] = torch.tensor(embedding).cuda()

        # 注意这里是对一个dataset中所有任务中的所有label ID都进行了处理和初始化。
        trainer.model_module.update_global_memory(memory_id_to_label_embedding)

    cfg.save_checkpoint_steps = trainer._optimizer.total_training_steps // cfg.epoch_num
    print("Base training, %d tasks, Train size: %d; Dev size: %d" % (len(base_task_keys), len(train_dataset), len(valid_dataset)))
    trainer.train()

    print("Updating local memories...")
    for task_key in base_task_keys:
        # 输入的dataset，有很多个task（meta-learning是以task为基本单位），找到这个task对应的所有label
        task_memory_label_ids = [valid_dataset.label_to_memory_id[label] for label in valid_dataset.task_to_label_mapping[task_key].keys()]
        # 输入：该task的ID和task对应的label ID （每个task对应多个ID）
        trainer.model_module.update_local_memory(valid_dataset.task_to_idx[task_key], task_memory_label_ids)
    trainer.save_checkpoint(save_best=True)

class LifelongTrainer(Trainer):
    def set_model_and_optimizer(self, model, cfg):
        self._model = model.to(self.cfg.local_rank)
        if self.cfg.n_gpu > 1:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model, device_ids=[self.cfg.local_rank],
                output_device=self.cfg.local_rank,
                find_unused_parameters=True)

        schedule = "warmup_linear"
        warmup_proportion = cfg.warmup_proportion
        max_grad_norm = cfg.max_grad_norm
        gradient_accumulation_steps = cfg.gradient_accumulation_steps
        num_steps_per_epoch = len(self._train_loader)
        epoch_num = cfg.epoch_num

        num_train_optimization_steps = int(math.ceil(num_steps_per_epoch / gradient_accumulation_steps * epoch_num))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        print("Slow LR: {}, Fast LR: {}".format(cfg.slow_lr, cfg.fast_lr))

        bert_model = self.model_module.bert if self.model_module.config.model_type == "bert" \
            else self.model_module.albert
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in bert_model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': cfg.slow_lr
             },
            {
                'params': [p for n, p in bert_model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': cfg.slow_lr
            },
            {
                'params': [p for n, p in [("memory", self.model_module.memory)] +
                           list(self.model_module.output.named_parameters())
                           if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': cfg.fast_lr
            },
            {
                'params': [p for n, p in [("memory", self.model_module.memory)] +
                           list(self.model_module.output.named_parameters())
                           if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': cfg.fast_lr
            },
        ]

        self._optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=cfg.learning_rate,
                             warmup=warmup_proportion,
                             max_grad_norm=max_grad_norm,
                             t_total=num_train_optimization_steps)


def meta_lifelong_training(cfg, other_task_keys):
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir_base, "final" if not cfg.use_replay else "final_aug")

    if cfg.use_lru:
        print("use LRU model")
        model = LRUMetaLabelEnhancedBertClassify.from_pretrained(
            os.path.join(cfg.checkpoint_dir_base, "base"),
            max_memory_size=cfg.max_memory_size,
            max_task_num=cfg.max_task_num,
            max_label_num=cfg.max_label_num,
            freeze_encoder=cfg.freeze_encoder)
    else:
        model = MetaLabelEnhancedBertClassify.from_pretrained(
            os.path.join(cfg.checkpoint_dir_base, "base"),
            max_memory_size=cfg.max_memory_size,
            max_task_num=cfg.max_task_num,
            max_label_num=cfg.max_label_num,
            freeze_encoder=cfg.freeze_encoder)

    if not io.isdir(os.path.join(cfg.checkpoint_dir_base, "final" if not cfg.use_replay else "final_aug")):
        os.makedirs(os.path.join(cfg.checkpoint_dir_base, "final" if not cfg.use_replay else "final_aug"))
    io.copy(os.path.join(os.path.join(cfg.checkpoint_dir_base, "base", "vocab.txt")),
            os.path.join(cfg.checkpoint_dir_base, "final" if not cfg.use_replay else "final_aug", "vocab.txt"))

    # Figure 1对应了一个high-level的图，两个过程，第一个过程对应的是初始化学习阶段->也就是main函数中的'base'条件判断
    # 第二个阶段对应的是lifelong 阶段，是在base阶段学习好以后，再进行学习的，main函数里面写了模型判断条件是整个框架图的过程。
    for task_index, task_key in enumerate(other_task_keys[:10]):
        print("[{}/{}] Task %s training...".format(task_index, len(other_task_keys), task_key))

        train_dataset = MetaIntentDataset(
            data_file=os.path.join(cfg.tables, task_key, "lifelong_task.json" if not cfg.use_replay else "train_aug.tsv"),
            meta_info_file=os.path.join(cfg.tables, "meta_info.json"),
            vocab_file=os.path.join(cfg.checkpoint_dir_base, "base", "vocab.txt"),
            max_seq_length=cfg.sequence_length,
            is_training=True)

        valid_dataset = MetaIntentDataset(
            data_file=os.path.join(cfg.tables, task_key, "lifelong_task.json"),
            meta_info_file=os.path.join(cfg.tables, "meta_info.json"),
            vocab_file=os.path.join(cfg.checkpoint_dir_base, "base", "vocab.txt"),
            max_seq_length=cfg.sequence_length,
            is_training=False)

        if cfg.freeze_encoder:
            trainer = Trainer(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, cfg=cfg)
        else:
            trainer = LifelongTrainer(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset, cfg=cfg)

        # with open(os.path.join(cfg.tables, task_key, "label_embeddings.json")) as f:
        #     label_embeddings = json.load(f)
        label_embeddings = load_label_emb(task_key)
        memory_id_to_label_embedding = dict()

        # 来了一个新任务了,新任务可能就是valid数据集，那么为了进行inference有local的memory，必须要得到他目前未知的label的global 表示，才能进行copy。
        # train的数据集是为了配合valid进行更新表示结果
        for label, embedding in label_embeddings.items():
            memory_id = valid_dataset.label_to_memory_id[label]
            memory_id_to_label_embedding[memory_id] = torch.tensor(embedding).cuda()

        trainer.model_module.update_global_memory(memory_id_to_label_embedding)

        cfg.save_checkpoint_steps = trainer._optimizer.total_training_steps // cfg.epoch_num
        # cfg.save_checkpoint_steps = 1
        trainer.train()
        # 最终是要取走这个task对应的valid数据集的memory进行inference
        task_memory_ids = [valid_dataset.label_to_memory_id[label] for label
                           in valid_dataset.task_to_label_mapping[task_key].keys()]
        trainer.model_module.update_local_memory(valid_dataset.task_to_idx[task_key], task_memory_ids)

def predict_meta_lifelong(cfg):
    if cfg.use_lru:
        print("use LRU model")
        model = LRUMetaLabelEnhancedBertClassify.from_pretrained(
            cfg.checkpoint_dir,
            max_memory_size=cfg.max_memory_size,
            max_task_num=cfg.max_task_num,
            max_label_num=cfg.max_label_num,
            freeze_encoder=True,
            is_testing=True).cuda()
    else:
        model = MetaLabelEnhancedBertClassify.from_pretrained(
            cfg.checkpoint_dir,
            max_memory_size=cfg.max_memory_size,
            max_task_num=cfg.max_task_num,
            max_label_num=cfg.max_label_num,
            freeze_encoder=True,
            is_testing=True).cuda()
    model.eval()

    test_dataset = MetaIntentDataset(
        data_file=os.path.join(cfg.tables, 'text_classify_49', "lifelong_task.json"),
        meta_info_file=os.path.join(cfg.tables, "meta_info.json"),
        vocab_file=os.path.join(cfg.checkpoint_dir_base, "base", "vocab.txt"),
        max_seq_length=cfg.sequence_length,
        max_label_num=cfg.max_label_num,
        is_training=False)

    testloader = DataLoader(test_dataset,
                            batch_size=cfg.eval_batch_size,
                            shuffle=False,
                            collate_fn=test_dataset.batch_fn)

    fout = io.open(cfg.outputs, "w")
    fout.write('pred_label' + "\t" + 'task' + "\t" + 'label' + "\n")
    print(len(testloader))
    for batch in tqdm(testloader):
        batch = {key: val.cuda() if isinstance(val, torch.Tensor) else val
                 for key, val in batch.items()}
        with torch.no_grad():
            model_outputs = model(batch)
        logits = model_outputs["logits"]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        label_ids = batch["label_ids"].tolist()

        tasks = batch["tasks"]

        for i, task in enumerate(tasks):
            label_mapping = test_dataset.task_to_label_mapping[task]
            idx_to_label = {idx: label for label, idx in label_mapping.items()}
            pred_label = idx_to_label[pred_ids[i]] if pred_ids[i] in idx_to_label else idx_to_label[0]
            label = idx_to_label[label_ids[i]]
            fout.write(pred_label + "\t" + task + "\t" + label + "\n")
    fout.close()
    print("Writing to %s finished. " % cfg.outputs)


def main(gpu, cfg, *args, **kwargs):
    init_running_envs(gpu, cfg)
    with io.open(os.path.join(cfg.tables, "base_tasks.json")) as f:
        base_tasks_keys = []
        base_tasks = eval(json.load(f))["data"]
        for item in base_tasks:
            base_tasks_keys.append(item['taskKey'])
        base_task_keys = set(base_tasks_keys)
    with io.open(os.path.join(cfg.tables, "meta_info.json")) as f:
        meta_info_list = eval(json.load(f))["data"]
    all_task_keys = set([t["taskKey"] for t in meta_info_list])
    other_task_keys = all_task_keys - base_task_keys

    # base task 有很多，选择base K个进行训练，如果没有设置，就默认所有的base task都进行训练
    if cfg.base_k:
        with io.open(os.path.join(cfg.tables, "base_tasks_%d.json" % cfg.base_k)) as f:
            base_task_keys = set(json.load(f))
    print('base task num: {}'.format(len(list(base_task_keys))))
    # logger.info("Base task Num: %d" % len(list(base_task_keys)))
    if cfg.mode == "train":
        if cfg.train_type == "base":
            base_mtl_training(cfg, base_task_keys)
        elif cfg.train_type == "lifelong":
            # other_task_keys: base就只是用来initialize一下
            meta_lifelong_training(cfg, sorted(other_task_keys))
        else:
            raise NotImplementedError
    elif cfg.mode == "predict":
        predict_meta_lifelong(cfg)

def load_label_emb(task_name):
    with open('MeLL_pytorch/data/label_dict.json', 'r') as file:
        all_tasks = eval(json.load(file))['data']
    task_labels = all_tasks[task_name]
    task_emb_dict = {}
    for task_label in task_labels:
        if task_label not in task_emb_dict.keys():
            task_emb_dict[task_label] = np.random.random(768)
    return task_emb_dict

if __name__ == "__main__":
    parser = config.add_basic_argument()
    parser = config.add_mtl_argument(parser)

    parser.add_argument("--checkpoint_dir_base", default='MeLL_pytorch/checkpoint', type=str, help="The base dir of checkpoints")
    parser.add_argument("--outputs", default='MeLL_pytorch/output/output_con.txt', type=str, help="The output Table")
    parser.add_argument("--max_memory_size", default=41, type=int, help="The max memory size")
    parser.add_argument("--slow_lr", default=1e-6, type=float, help="The slow learning rate")
    parser.add_argument("--fast_lr", default=1e-3, type=float, help="The fast learning rate")
    parser.add_argument("--train_type", default="lifelong", type=str, choices=["base", "lifelong"],
                        help="which part of model will be trained")
    parser.add_argument("--use_replay", default=False, action="store_true", help='Using replay')
    parser.add_argument("--use_lru", default=True, help='Using LRU memory')
    parser.add_argument("--base_k", default=None, type=int, help='Only select k tasks in base tasks')
    cfg = parser.parse_args()
    distributed_call_main(main_fn=main, cfg=cfg)
