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

# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:10 pm
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @Github  : https://github.com/alibaba/EasyTransfer, https://github.com/wjn1996

import ast
import json
import os
import statistics
from collections import defaultdict
from typing import List, Dict
import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import log
from pet.config import EvalConfig, TrainConfig
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.wrapper import TransformerModelWrapper
from pet.transprompt_wrapper import TransPromptModelWrapper
from pet.transprompt_wrapper2 import TransPromptModelWrapper as TransPromptModelWrapper2
from pet.config import  WrapperConfig
from data_utils.task_processors import  PROCESSORS, load_examples, DEV32_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS, SPE_TRAIN_SET, SPE_DEV_SET
from data_utils.utils import groups, data_to_name
logger = log.get_logger('root')




def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    ## edit by wjn
    ## 如果当前任务是single task，则获取single task对应的model
    if config.task_type is None or config.task_type == 'single_task':
        model = TransformerModelWrapper(config)
    else: # 对于多任务情况下，则获取cross-task
        model = TransPromptModelWrapper(config)
    return model


# edit by wjn 取消了多次重复实验
def train_pet(train_data: List[InputExample],
              eval_data: List[InputExample],
              dev32_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              seed: int = 42
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    # set_seed(seed)

    assert model_config.task_type == "single_task"

    for pattern_id in pattern_ids: # pattern只有1个

        model_config.pattern_id = pattern_id
        results_dict = {}

        pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, 1)

        # if os.path.exists(pattern_iter_output_dir):
        #     logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
        #     continue

        if not os.path.exists(pattern_iter_output_dir):
            os.makedirs(pattern_iter_output_dir)

        wrapper = init_model(model_config) # 初始化一个模型

        # Training
        if do_train:
            # 开始多轮epoch训练，并将训练的结果保存到results_dict中
            results_dict.update(train_single_model(train_data, eval_data, dev32_data, pattern_iter_output_dir, \
                                                   wrapper, train_config, eval_config))

            with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                fh.write(str(results_dict))

            train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
            eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
            logger.info("Saving complete")

            if not do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

        # Evaluation
        if do_eval:
            logger.info("Starting evaluation...")
            logger.info("Single: Task {} 's Test examples number: {}".format(model_config.task_name, len(eval_data)))

            logger.info("************Test Example:**************")
            logger.info("text_a={}".format(eval_data[0].text_a))
            logger.info("text_b={}".format(eval_data[0].text_b))
            logger.info("task={}".format(eval_data[0].task))
            logger.info("label={}".format(eval_data[0].label))
            logger.info("**********************************")

            # if not wrapper:
            wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

            eval_result = evaluate(wrapper, eval_data, eval_config)
            # dev32_result = evaluate(wrapper, dev32_data, eval_config)

            save_predictions(os.path.join(pattern_iter_output_dir, 'eval_predictions.jsonl'), wrapper, eval_result)
            save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

            # save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_result)
            # save_logits(os.path.join(pattern_iter_output_dir, 'dev32_logits.txt'), dev32_result['logits'])

            logger.info("--- RESULT (pattern_id={}, Task={}) ---".format(pattern_id, model_config.task_name))
            logger.info("eval_results:")
            logger.info(eval_result['scores'])
            # logger.info("dev32_results:")
            # logger.info(dev32_result['scores'])

            # results_dict['eval_set_after_training'] = eval_result['scores']
            # results_dict['dev32_set_after_training'] = dev32_result['scores']
            # with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
            #     json.dump(results_dict, fh)
            #
            # for metric, value in eval_result['scores'].items():
            #     results[metric][pattern_id].append(value)
            #
            # for metric, value in dev32_result['scores'].items():
            #     dev32_results[metric][pattern_id].append(value)

            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()

    # if do_eval:
    #     logger.info("=== OVERALL RESULTS ===")
    #     _write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
    # else:
    #     logger.info("=== ENSEMBLE TRAINING COMPLETE ===")





### add by wjn
### 用于cross-task的训练和测试
### 输入cross-task的训练集和验证集进行训练，在测试时，将当前group内的所有数据加载进来，并分别进行测试

def train_pet_cross(train_data: List[InputExample],
              # eval_data: List[InputExample],
              dev32_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              seed: int = 42
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    # set_seed(seed)

    assert model_config.task_type == "cross_task"
    # 当前是cross-task，则task_name是group的名称，需要获得group内的所有task
    tasks = groups[model_config.task_name]

    for pattern_id in pattern_ids: # 只选择1个模式

        model_config.pattern_id = pattern_id
        results_dict = {}

        pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, 1)

        # if os.path.exists(pattern_iter_output_dir):
        #     logger.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
        #     continue

        if not os.path.exists(pattern_iter_output_dir):
            os.makedirs(pattern_iter_output_dir)

        wrapper = init_model(model_config) # 初始化一个模型

        # Training
        if do_train:
            # 开始多轮epoch训练，并将训练的结果保存到results_dict中
            # edit by wjn : eval_data -> None
            results_dict.update(train_single_model(train_data, None, dev32_data, pattern_iter_output_dir, \
                                                   wrapper, train_config, eval_config))

            with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                fh.write(str(results_dict))

            train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
            eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
            logger.info("Saving complete")

            if not do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

        # Evaluation
        if do_eval:
            logger.info("Starting evaluation...")

            # if not wrapper:
            wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
            cross_data_dir = "data/k-shot-cross/"
            # add by wjn
            ## 当前是cross-task，对当前group内的所有task，分别进行测试
            for task_name in tasks:
                eval_data = load_examples(
                    task_name, cross_data_dir + data_to_name[task_name] + "/" + str(model_config.k) + "-" + str(seed),
                    TEST_SET, num_examples=-1, num_examples_per_label=None)
                logger.info("Group {}: Task {} 's Test examples number: {}".format(model_config.task_name, task_name, len(eval_data)))

                logger.info("************Test Example:**************")
                logger.info("text_a={}".format(eval_data[0].text_a))
                logger.info("text_b={}".format(eval_data[0].text_b))
                logger.info("task={}".format(eval_data[0].task))
                logger.info("label={}".format(eval_data[0].label))
                logger.info("**********************************")

                # 更新当前group task的metrics：
                eval_config.metrics = METRICS.get(task_name, DEFAULT_METRICS) # cross-task group 的 metrics
                eval_result = evaluate(wrapper, eval_data, eval_config)
            # dev32_result = evaluate(wrapper, dev32_data, eval_config)

                save_predictions(os.path.join(pattern_iter_output_dir, 'eval_predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                # save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_result)
                # save_logits(os.path.join(pattern_iter_output_dir, 'dev32_logits.txt'), dev32_result['logits'])

                logger.info("--- RESULT (pattern_id={}, Group={}, Task={}) ---".format(pattern_id, model_config.task_name, task_name))
                logger.info("eval_results:")
                logger.info(eval_result['scores'])
                # logger.info("dev32_results:")
                # logger.info(dev32_result['scores'])

            # results_dict['eval_set_after_training'] = eval_result['scores']
            # # results_dict['dev32_set_after_training'] = dev32_result['scores']
            # with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
            #     json.dump(results_dict, fh)
            #
            # for metric, value in eval_result['scores'].items():
            #     results[metric][pattern_id].append(value)
            #
            # for metric, value in dev32_result['scores'].items():
            #     dev32_results[metric][pattern_id].append(value)

            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()

    # if do_eval:
    #     logger.info("=== OVERALL RESULTS ===")
    #     _write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
    # else:
    #     logger.info("=== ENSEMBLE TRAINING COMPLETE ===")


### add by wjn
### Task Adaptation
### 首先加载cross-task的训练集和验证集，进行训练和验证，并保存模型；
### 其次分别在各自的task上再次进行微调训练和验证；
### 在对某一个task微调后，在对应测试集上预测

def train_adaptation_cross(train_data: List[InputExample],
              # eval_data: List[InputExample],
              dev32_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              seed: int = 42
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    assert model_config.task_type == "cross_task"
    # 当前是cross-task，则task_name是group的名称，需要获得group内的所有task
    tasks = groups[model_config.task_name]

    for pattern_id in pattern_ids: # 只选择1个模式

        model_config.pattern_id = pattern_id
        results_dict = {}

        pattern_iter_output_dir = "{}/{}/adaptation/{}".format(output_dir, model_config.scene, model_config.task_name)

        

        if not os.path.exists(pattern_iter_output_dir):
            os.makedirs(pattern_iter_output_dir)

        # wrapper = TransPromptModelWrapper(model_config) # 初始化一个TransPrompt模型
        wrapper = TransPromptModelWrapper2(model_config)  # 初始化一个TransPrompt模型
        # wrapper = TransformerModelWrapper(model_config)

        # Multi-Task Meta-Learning Training
        if do_train:

            logger.info("========= Stage1: Starting Fine-tuning Multi-Task Meta-Learner ... =========")
            # 开始多轮epoch训练，并将训练的结果保存到results_dict中
            # edit by wjn : eval_data -> None
            results_dict.update(train_single_model(train_data, None, dev32_data, pattern_iter_output_dir, \
                                                   wrapper, train_config, eval_config, use_debias=True))


            train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
            eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
            logger.info("Saving complete")

            if not do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

            logger.info("========= Stage1: Finish Fine-tuning Multi-Task Meta-Learner =========")

        # Task Adaptation Fine-tune
        if do_eval:
            logger.info("========= Stage2: Starting Task Adaptation (Task-Specific Fine-tuning) ... =========")
            # 用于保存每次试验跑出的结果
            # 加载先前的结果
            t = time.time()
            ada_res_acc = dict()
            if os.path.exists('ada_res_acc.npy'):
                ada_res_acc = np.load('ada_res_acc.npy', allow_pickle=True)[()] # dict {time: {task:acc, ...}}
            accs = dict()
            # 重新加载训练好的meta-learner
            # wrapper = TransPromptModelWrapper.from_pretrained(pattern_iter_output_dir)
            wrapper = TransPromptModelWrapper2.from_pretrained(pattern_iter_output_dir)
            cross_data_dir = "data/k-shot-cross/"
            # add by wjn
            ## 当前是task adaptation，对每个group的每个task:
            # 在训练好的meta learner基础上，在当前task对应的训练集上再次task specific fine-tune；并在验证集上选择模型
            # 最后在对应的测试集上测试；
            # group内的每个task在此阶段独立地进行

            # 获得cross-task上每个task对应的训练集和测试集
            task_to_train_example, task_to_dev_example = dict(), dict()  # {task_name: [.., ..], ..}
            task_to_train_example = load_examples(
                model_config.task_name, None, SPE_TRAIN_SET, num_examples=-1, num_examples_per_label=None, examples=train_data)
            task_to_dev_example = load_examples(
                model_config.task_name, None, SPE_DEV_SET, num_examples=-1, num_examples_per_label=None, examples=dev32_data)


            for ei, task_name in enumerate(tasks):

                ### task-specific fine-tune
                logger.info("========= Stage2.{}: Specific fine-tuning on Task {} =========".format(ei + 1, task_name))
                # wrapper.config.task_name = task_name # 在task-specific微调时，更改当前微调的task名称
                train_config.max_steps = eval_config.max_steps # 在task-specific微调时，更改max_steps
                train_config.per_gpu_train_batch_size = eval_config.per_gpu_eval_batch_size # 更改batch_size
                if task_name == 'mrpc': # group3内的两个task（MRPC和QQP）训练集样本数量悬殊过大，直接指定MRPC只有1200steps
                    train_config.max_steps = 4800
                    train_config.per_gpu_train_batch_size = 16
                    eval_config.per_gpu_eval_batch_size = 8
                # 在meta-learner基础上继续做task-specific微调，并保存
                train_single_model(task_to_train_example[data_to_name[task_name]], None,
                                   task_to_dev_example[data_to_name[task_name]], pattern_iter_output_dir + '/' + task_name, \
                                   wrapper, train_config, eval_config, use_debias=False)
                # 将task-specific微调后保存的模型加载进来
                # task_specific_wrapper = TransPromptModelWrapper.from_pretrained(pattern_iter_output_dir + '/' + task_name)
                task_specific_wrapper = TransPromptModelWrapper2.from_pretrained(pattern_iter_output_dir + '/' + task_name)
                logger.info("========= Stage2.{}: Evaluating test set on Task {}".format(ei + 1, task_name))
                ### evaluate on test dataset
                eval_data = load_examples(
                    task_name, cross_data_dir + data_to_name[task_name] + "/" + str(model_config.k) + "-" + str(seed),
                    TEST_SET, num_examples=-1, num_examples_per_label=None)
                logger.info("Group {}: Task {} 's Test examples number: {}".format(model_config.task_name, task_name, len(eval_data)))

                # logger.info("************Test Example:**************")
                # logger.info("text_a={}".format(eval_data[0].text_a))
                # logger.info("text_b={}".format(eval_data[0].text_b))
                # logger.info("task={}".format(eval_data[0].task))
                # logger.info("label={}".format(eval_data[0].label))
                # logger.info("**********************************")

                # 更新当前group task的metrics：
                eval_config.metrics = METRICS.get(task_name, DEFAULT_METRICS) # cross-task group 的 metrics
                eval_result = evaluate(task_specific_wrapper, eval_data, eval_config)

                save_predictions(os.path.join(pattern_iter_output_dir + '/' + task_name, 'eval_predictions.jsonl'), task_specific_wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir + '/' + task_name, 'eval_logits.txt'), eval_result['logits'])

                # save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_result)
                # save_logits(os.path.join(pattern_iter_output_dir, 'dev32_logits.txt'), dev32_result['logits'])

                logger.info("--- Task Adaptation Result (pattern_id={}, Group={}, Task={}) ---".format(pattern_id, model_config.task_name, task_name))
                logger.info("eval_results: {}".format(eval_result['scores']))
                accs[task_name] = eval_result['scores']
                task_specific_wrapper.model = None
                task_specific_wrapper = None
            ada_res_acc[t] = accs
            np.save('ada_res_acc.npy', ada_res_acc)
            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()


    # if do_eval:
    #     logger.info("=== OVERALL RESULTS ===")
    #     _write_results(os.path.join(output_dir, 'result_test.txt'), results, dev32_results)
    # else:
    #     logger.info("=== ENSEMBLE TRAINING COMPLETE ===")




### add by wjn
### Task Generalization
### 首先加载cross-task的训练集和验证集，进行训练和验证，并保存模型；
### 其次分别在各自的task上再次进行微调训练和验证；
### 在对某一个task微调后，在对应测试集上预测

def train_generalization_cross(unseen_task_train_data: List[InputExample],
              unseen_task_dev_data: List[InputExample],
              seen_task_train_data: List[InputExample],
              seen_task_dev_data: List[InputExample],
              # dev32_data: List[InputExample],
              unseen_task: str,
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              pattern_ids: List[int],
              output_dir: str,
              repetitions: int = 3,
              do_train: bool = True,
              do_eval: bool = True,
              seed: int = 42
              ):

    """
    Train and evaluate a new PET model for a given task.

    :param model_config: the model configuration for each model corresponding to an individual PVP
    :param train_config: the training configuration for each model corresponding to an individual PVP
    :param eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param train_data: the training examples to use
    :param dev32_data: the dev32 examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    dev32_results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    assert model_config.task_type == "cross_task"

    for pattern_id in pattern_ids: # 只选择1个模式

        model_config.pattern_id = pattern_id
        results_dict = {}

        pattern_iter_output_dir = "{}/{}/generalization/{}".format(output_dir, model_config.scene, model_config.task_name)

        if not os.path.exists(pattern_iter_output_dir):
            os.makedirs(pattern_iter_output_dir)

        # wrapper = TransPromptModelWrapper(model_config) # 初始化一个TransPrompt模型
        wrapper = TransPromptModelWrapper2(model_config) # 初始化一个TransPrompt模型
        # wrapper = TransformerModelWrapper(model_config)

        # Multi-Task Meta-Learning Training
        if do_train:

            logger.info("========= Stage1: Starting Fine-tuning Multi-Task Meta-Learner ... =========")
            # 开始多轮epoch训练，并将训练的结果保存到results_dict中
            # edit by wjn : eval_data -> None
            results_dict.update(train_single_model(seen_task_train_data, None, seen_task_dev_data, pattern_iter_output_dir, \
                                                   wrapper, train_config, eval_config, use_debias=True))


            train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
            eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
            logger.info("Saving complete")

            if not do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

            logger.info("========= Stage1: Finish Fine-tuning Multi-Task Meta-Learner =========")

        # Task Adaptation Fine-tune
        if do_eval:
            logger.info("========= Stage2: Starting Task Generalization (Unseen Task-Specific Fine-tuning) ... =========")

            # 重新加载训练好的meta-learner
            # wrapper = TransPromptModelWrapper.from_pretrained(pattern_iter_output_dir)
            wrapper = TransPromptModelWrapper2.from_pretrained(pattern_iter_output_dir)
            cross_data_dir = "data/k-shot-cross/"
            # add by wjn
            ## 当前是task generalization，对每个group的每个task:
            # 在训练好的meta learner基础上，在当前task对应的训练集上再次task specific fine-tune；并在验证集上选择模型
            # 最后在对应的测试集上测试；
            # group内的每个task在此阶段独立地进行


            ### task-specific fine-tune
            logger.info("========= Stage2: Specific fine-tuning on Unseen Task {} =========".format(unseen_task))
            # wrapper.config.task_name = task_name # 在task-specific微调时，更改当前微调的task名称
            train_config.max_steps = eval_config.max_steps # 在task-specific微调时，更改max_steps
            train_config.per_gpu_train_batch_size = eval_config.per_gpu_eval_batch_size # 更改batch_size

            # 在meta-learner基础上对unseen task继续做task-specific微调，并保存
            train_single_model(unseen_task_train_data, None,
                               unseen_task_dev_data, pattern_iter_output_dir + '/' + unseen_task, \
                               wrapper, train_config, eval_config, use_debias=False)
            # 将task-specific微调后保存的模型加载进来
            # task_specific_wrapper = TransPromptModelWrapper.from_pretrained(pattern_iter_output_dir + '/' + unseen_task)
            task_specific_wrapper = TransPromptModelWrapper2.from_pretrained(pattern_iter_output_dir + '/' + unseen_task)
            logger.info("========= Stage2: Evaluating test set on Task {}".format(unseen_task))
            ### evaluate on test dataset
            eval_data = load_examples(
                unseen_task, cross_data_dir + data_to_name[unseen_task] + "/" + str(model_config.k) + "-" + str(seed),
                TEST_SET, num_examples=-1, num_examples_per_label=None)
            logger.info("Group {}: Task {} 's Test examples number: {}".format(model_config.task_name, unseen_task, len(eval_data)))

            # logger.info("************Test Example:**************")
            # logger.info("text_a={}".format(eval_data[0].text_a))
            # logger.info("text_b={}".format(eval_data[0].text_b))
            # logger.info("task={}".format(eval_data[0].task))
            # logger.info("label={}".format(eval_data[0].label))
            # logger.info("**********************************")

            # 更新当前group task的metrics：
            eval_config.metrics = METRICS.get(unseen_task, DEFAULT_METRICS) # cross-task group 的 metrics
            eval_result = evaluate(task_specific_wrapper, eval_data, eval_config)

            save_predictions(os.path.join(pattern_iter_output_dir + '/' + unseen_task, 'eval_predictions.jsonl'), task_specific_wrapper, eval_result)
            save_logits(os.path.join(pattern_iter_output_dir + '/' + unseen_task, 'eval_logits.txt'), eval_result['logits'])

            # save_predictions(os.path.join(pattern_iter_output_dir, 'dev32_predictions.jsonl'), wrapper, dev32_result)
            # save_logits(os.path.join(pattern_iter_output_dir, 'dev32_logits.txt'), dev32_result['logits'])

            logger.info("--- Unseen Task Generalization Result (pattern_id={}, Group={}, Task={}) ---".format(pattern_id, model_config.task_name, unseen_task))
            logger.info("eval_results: {}".format(eval_result['scores']))

            task_specific_wrapper.model = None
            task_specific_wrapper = None

            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()






# 训练
def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev32_data: List[InputExample],
                       pattern_iter_output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig,
                       use_debias: bool = False):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}

    # results_dict['train_set_before_training'] = evaluate(model, train_data, eval_config)['scores']['acc']

    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            pattern_iter_output_dir=pattern_iter_output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev32_data=dev32_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha,
            use_debias=use_debias
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    # 加载训练好的模型
    # model = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
    # results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config)['scores']['acc']
    return results_dict


def evaluate(model: TransformerModelWrapper,
             eval_data: List[InputExample],
             config: EvalConfig) -> Dict:

    metrics = config.metrics if config.metrics else ['acc']
    results = model.eval(eval_data=eval_data,
                         per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu)
    # print("results['logits'].shape=", results['logits'].shape)
    predictions = np.argmax(results['logits'], axis=1)
    scores = {}
    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")
    results['scores'] = scores
    results['predictions'] = predictions
    return results


def _write_results(path: str, all_results: Dict, dev32_results: Dict):
    with open(path, 'w') as fh:

        results = all_results
        logger.info("eval_results:")
        fh.write("eval_results:" + '\n')

        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

        logger.info("dev32_results:")
        fh.write("dev32_results:" + '\n')

        for metric in dev32_results.keys():
            for pattern_id, values in dev32_results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logger.info(result_str)
                fh.write(result_str + '\n')

        for metric in dev32_results.keys():
            all_results = [result for pattern_results in dev32_results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logger.info(result_str)
            fh.write(result_str + '\n')

