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

import logging
import os
import re
import subprocess as sp
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import List, Tuple
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
yaml_load = partial(load, Loader=Loader)


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = ArgumentParser("Experiment Launcher")
parser.add_argument("--devices", "-v", type=str, default=None)
parser.add_argument("--tasks", "-t", type=int, nargs="+", default=None)
parser.add_argument("--debug", "-g", action="store_true")
parser.add_argument("--model_type", type=str, default="roberta_large")
parser.add_argument("--cross_k", type=int, default=160)
parser.add_argument("-k", type=int, default=16)
parser.add_argument("-s", type=int, dest="seed", default=13)
options = parser.parse_args()


def load_task_settings():
    """Settings for each task, such as template and tag mapping.
    This should be consistent across all scripts, so it is stored in a YAML file.
    """
    with open("task_settings.yaml", "r") as f:
        task_settings = yaml_load(f)

    def _get_task_list() -> List[str]:
        return list(task_settings.keys())

    def _get_task_settings(task_name: str) -> Tuple[str, ...]:
        settings = task_settings[task_name]
        return (
            settings.get("template"),
            settings.get("mapping"),
            settings.get("extra"),
        )

    return _get_task_list, _get_task_settings


get_task_list, get_task_settings = load_task_settings()

if options.devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = options.devices
    logger.info(f'Visible CUDA devices: {os.environ["CUDA_VISIBLE_DEVICES"]}')
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    logger.info("Visible CUDA devices: all")

py_exec = [sys.executable]
if options.debug:
    py_exec.extend("-m debugpy --listen 5678 --wait-for-client".split())
    logger.info("Running in debug mode")
else:
    logger.info("Running in normal mode")
py_exec.append("cli.py")


def run_task(args) -> None:
    task_name = re.findall(r"--task_name=(.+?)\s+", args)[0]
    out_dir = re.findall(r"--output_dir=(.+?)\s+", args)[0]

    logger.info(f"Start to process task {task_name}")

    py_exec.append(args)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    with out_dir.joinpath("run.log").open("w") as log_fp:
        cmd = " ".join(py_exec)
        proc = sp.run(cmd, shell=True, stdout=log_fp, stderr=sp.STDOUT)
    try:
        proc.check_returncode()
    except sp.CalledProcessError:
        logger.error(f"Task {task_name} FAILED")
        logger.error(f'Please check the log file: {out_dir / "run.log"}')
    else:
        logger.info(f"Task {task_name} SUCCEED")

def read_log_file(path):
    try:
        with open(path, "r") as f:
            contexts = f.readlines()
    except:
        return f"{0}"
    acc = round(float(contexts[1].strip().split(" ")[2]),4) * 100
    return f"{acc:.2f}"

def read_task_res(task, suffix, k, seed=None):
    assert task in ["mnli", "snli", "sst-2", "mr", "mrpc", "qqp", "qnli", "rte"]
    if seed is None:
        seed = [13, 21, 42, 87, 100]
    root_path = "./results"
    res = [task]
    if "student" in suffix:
        path = os.path.join(root_path, task, suffix, "bert_small")
    else:
        path = os.path.join(root_path, task, suffix, "roberta_large")
    for one_seed in seed:
        path = os.path.join(path, f"{k}-{one_seed}", f"test_results_{task}")
        res.append(read_log_file(path))
    if task != "mnli":
        return res
    else:
        res_mnli_mm = [f"{task}-mm"]
        for one_seed in seed:
            path = os.path.join(path, f"{k}-{one_seed}", f"test_results_{task}-mm")
            res_mnli_mm.append(read_log_file(path))
    return res, res_mnli_mm