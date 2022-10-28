# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 2:52 pm.
# @Author  : JianingWang
# @File    : config.py
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from transformers import MODEL_FOR_MASKED_LM_MAPPING
from transformers import TrainingArguments as TransformersTrainingArguments

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    from_scratch: bool = field(
        default=False,
        metadata={'help': 'from scratch'}
    )

    early_stopping_patience: int = field(
        default=None,
        metadata={'help': 'early_stopping_patience for early stopping callback'}
    )

    freeze_epochs: Optional[float] = field(
        default=None,
        metadata={
            'help': '对前n个epoch进行冻层操作, float类型如 0.5'
        }
    )
    freeze_keyword: Optional[str] = field(
        default='encoder',
        metadata={
            'help': '包含keyword的layer将会被冻层'
        }
    )

    ema: bool = field(
        default=False,
        metadata={'help': '是否使用ema'}
    )

    ema_decay: float = field(
        default=0.999,
        metadata={'help': 'ema系数'}
    )
    do_lower_case: bool = field(
        default=False,
        metadata={'help': 'do_lower_case'}
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the task."}
    )
    task_type: Optional[str] = field(
        default='classification', metadata={"help": "任务类型：classification, mlm"}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "数据路径"}
    )
    exp_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the experiment."}
    )
    tracking_uri: Optional[str] = field(
        default=None, metadata={"help": "The uri of mlflow."}
    )
    mlflow_location: Optional[str] = field(
        default=None, metadata={"help": "The location of mlflow tracking and artifact"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    max_eval_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization for eval and test set. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )

    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    cross_validation_num: Optional[int] = field(
        default=None,
        metadata={'help': 'cross_validation_num'}
    )

    # Question Answer
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                    "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    keep_predict_labels: Optional[bool] = field(
        default=False, metadata={"help": "keep_predict_labels"}
    )
    post_tokenizer: Optional[bool] = field(
        default=False, metadata={"help": "post_tokenizer"}
    )
    user_defined: Optional[str] = field(
        default="", metadata={"help": "User defined by you-self, split by '\space' e.g. 'name=xxx year=2000'"}
    )

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
        return d


@dataclass
class TrainingArguments(TransformersTrainingArguments):
    do_adv: bool = field(
        default=False,
        metadata={"help": "do fgm adversarial attack."},
    )
    do_predict_during_train: bool = field(
        default=False
    )


