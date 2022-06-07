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

import torch
import torch.nn.functional as F
import os
import numpy as np
import logging
import scipy.spatial.distance as distance

from running_utils import get_task_list, options

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


task_list_map = get_task_list()
task_list = [task_list_map[task] for task in options.tasks]
logger.info(f"Task: {task_list}")

assistant_model_mapping = {
    "sst-2": "mr",
    "mr": "sst-2",
    "mrpc":"qqp",
    "qqp": "mrpc",
    "mnli": "snli",
    "snli": "mnli",
    "qnli": "rte",
    "rte": "qnli",
}

def normalization_cosine_similarity(logit1, logit2, dim=-1):
    return 0.5 + 0.5 * torch.cosine_similarity(logit1, logit2, dim=dim)

def normalization_euclidean_distance(logit1, logit2):
    return torch.sqrt(torch.sum((logit1 - logit2) ** 2))

def jensen_shannon_divergence(logit1, logit2, softmax=True):
    if softmax:
        logit1 = F.softmax(logit1, dim=-1)
        logit2 = F.softmax(logit2, dim=-1)
    log_mean = ((logit1 + logit2) / 2).log()
    return 1 - (F.kl_div(log_mean, logit1, reduction="batchmean") + F.kl_div(log_mean, logit2, reduction="batchmean"))

def cosine_similarity(arr1, arr2):
    return 1 - distance.cosine(arr1, arr2)

def compute_center(arr):
    return np.mean(arr, axis=0)

seed_list = [options.seed]
k_list = [options.k]
suffix = "save_logits"
ood_task = ["sst-2", "mr"]

def main():
    # Compute Weight
    for task in task_list:
        for seed in seed_list:
            for k in k_list:
                if task not in ood_task and k in [64, 112]:
                    continue
                embedding_path = os.path.join("./results", task, suffix, options.model_type, f"{k}-{seed}")
                embedding_in = np.load(os.path.join(embedding_path, "cls_logits.pkl"))
                embedding_cross = np.load(os.path.join(embedding_path, "cls_logits_ass.pkl"))
                assert embedding_in.shape == embedding_cross.shape
                weights = list()
                for embed_in, embed_cross in zip(embedding_in, embedding_cross):
                    weights.append(jensen_shannon_divergence(torch.tensor(embed_in), torch.tensor(embed_cross)).numpy())
                weight_out_path = os.path.join("./results", task, "ptkd_teacher", "roberta_large", f"{options.cross_k}-{seed}")
                with open(os.path.join(weight_out_path, "weights.pkl"), "wb") as f:
                    np.save(f, weights)
                logger.info(f"{task} weights saved in {weight_out_path}")

if __name__ == "__main__":
    main()