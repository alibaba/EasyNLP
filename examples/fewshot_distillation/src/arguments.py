# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and Princeton Natural Language Processing.
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

from dataclasses import dataclass, field
from tkinter.messagebox import NO
from typing import Optional

from transformers.training_args import TrainingArguments
from transformers.data.datasets.glue import GlueDataTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default="prompt",
        metadata={
            "help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"
        },
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={
            "help": "Whether to reinitialize the token type embeddings (only for BERT)."
        },
    )


@dataclass
class DynamicDataTrainingArguments(GlueDataTrainingArguments):
    """
    Arguments for dynamic training.
    """

    num_k: Optional[int] = field(
        default=16, metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={
            "help": "Number of samples (for inference) in fine-tuning with demonstrations"
        },
    )

    num_demo: Optional[int] = field(
        default=1, metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"},
    )

    # For prompting
    template: str = field(default=None, metadata={"help": "Template"})

    mapping: str = field(default=None, metadata={"help": "Label word mapping"})

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"
        },
    )

    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"
        },
    )

    prompt_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"
        },
    )

    template_id: int = field(
        default=None, metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None, metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None, metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None, metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default="",
        metadata={"help": "Set the tag and find the result easier in the log."},
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False, metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"},
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."
        },
    )

    debug_mode: bool = field(default=False, metadata={"help": "Debug mode"})

    # For max length
    double_demo: bool = field(
        default=False, metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"},
    )

    other_sent_limit: int = field(
        default=None,
        metadata={
            "help": "Limit the length of sentences other than the first sentence"
        },
    )

    use_full_length: bool = field(
        default=None, metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"},
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"},
    )

    gpt3_in_context_num: int = field(
        default=32, metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={
            "help": "When exceeding the maximum length, truncate the head instead of the tail."
        },
    )

    # **********Adding Change **********
    with_weight: bool = field(
        default=False,
        metadata={"help": "Whether to use meta-weights fine-tuning"}
        )
    
    with_inter: bool = field(
        default=False,
        metadata={"help": "Whether to use inter_logits fine-tuning"}
        )
    
    multi: bool = field(
        default=False,
        metadata={"help": "Whether to export 10x data logits"}
        )
    
    with_high_prob: bool = field(
        default=False,
        metadata={"help": "Whether to use manual higher accuracy prob"}
        )
    # ********** END **********
    
    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False, metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={
            "help": "(DO NOT List of templates (only initialized after the program starts."
        },
    )
    


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={
            "help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"
        },
    )

    model_id: int = field(
        default=-1,
        metadata={
            "help": "Model ID (contains template information) to identify the model"
        },
    )

    save_logit: bool = field(
        default=False,
        metadata={
            "help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"
        },
    )

    save_logit_dir: str = field(
        default=None, metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0, metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={
            "help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"
        },
    )

    # Turn off train/test
    no_train: bool = field(default=False, metadata={"help": "No training"})
    no_eval: bool = field(default=False, metadata={"help": "No training"})
    no_predict: bool = field(default=False, metadata={"help": "No test"})
    do_debug: bool = field(default=False)

    # Distillation
    teacher_mode: bool = field(default=False)
    student_mode: bool = field(default=False)
    zero_shot: bool = field(default=False)
    export_embedding: bool = field(default=False)

    temperature: float = field(default=1.0)
    alpha: float = field(default=0.0)
    beta: float = field(default=0.0)
    gamma: float = field(default=0.0)

    def __post_init__(self):
        super().__post_init__()
        if self.no_train:
            self.do_train = False
        if self.no_eval:
            self.do_eval = False
        if self.no_predict:
            self.do_predict = False
        if self.save_logit_dir is None:
            self.save_logit_dir = self.output_dir

        if self.export_embedding:
            assert self.teacher_mode


if __name__ == "__main__":
    print(0)
