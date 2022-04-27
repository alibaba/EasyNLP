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
"""EasyNLP arguments."""

import argparse
import importlib
import os


def is_torchx_available():
    return importlib.util.find_spec('torchacc') is not None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def parse_args(extra_args_provider=None,
               defaults={},
               ignore_unknown_args=False) -> argparse.ArgumentParser:
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='EasyNLP Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_easynlp_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print(
                    'WARNING: overriding default arguments for {key}:{v} with {key}:{v2}'
                    .format(key=key, v=defaults[key], v2=getattr(args, key)),
                    flush=True)
        else:
            setattr(args, key, defaults[key])

    assert args.mode is not None
    assert args.tables is not None

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0

    if 'odps://' in args.tables:
        args.read_odps = True
    else:
        args.read_odps = False

    # args.n_gpu = args.worker_gpu if args.worker_gpu > 0 else 0
    if is_torchx_available():
        args.n_gpu = 0
    else:
        args.n_gpu = args.world_size
    args.n_cpu = args.worker_cpu if args.worker_cpu > 0 else 1
    if args.rank == 0:
        args.is_master_node = True
    else:
        args.is_master_node = False
    _print_args(args)
    return args


def parse_args_for_cli(extra_args_provider=None,
                       defaults={},
                       ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='EasyNLP CLI Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_easynlp_args(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    args.rank = int(os.getenv('RANK', '0'))
    # _print_args_cli(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)


def _print_args_cli(args):
    """Print arguments."""
    if args.rank == 0:
        print(
            '------------------------ cli arguments ------------------------',
            flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(
            '-------------------- end of cli arguments ---------------------',
            flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_easynlp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='EasyNLP')

    group.add_argument('--random_seed',
                       type=int,
                       default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')

    group.add_argument('--mode',
                       default='train',
                       type=str,
                       choices=['train', 'evaluate', 'predict'],
                       help='The mode of easynlp')

    group.add_argument('--user_script',
                       default=None,
                       type=str,
                       help='The scripts to run with easynlp')

    group.add_argument('--user_entry_file',
                       default=None,
                       type=str,
                       help='The entry file of the scripts')

    group.add_argument('--tables',
                       default=None,
                       type=str,
                       help='The input table, '
                       '`train_file`,`valid_file` if mode=`train`;'
                       '`valid_file` if mode=`evaluate`')

    group.add_argument(
        '--user_defined_parameters',
        default=None,
        type=str,
        help='user_defined_parameters specified by -DuserDefinedParameters')

    group.add_argument('--skip_first_line',
                       action='store_true',
                       help='Whether to skip the first line in data files.')

    group.add_argument('--outputs',
                       default=None,
                       type=str,
                       help='The output table, '
                       'output prediction file')

    group.add_argument('--buckets', type=str, default=None, help='Oss buckets')

    group.add_argument('--odps_config',
                       type=str,
                       default=None,
                       help='Config file path of odps')

    group.add_argument('--app_name',
                       default='text_classify',
                       type=str,
                       choices=[
                           'text_classify', 'text_classify_multi_label',
                           'text_match', 'text_match_two_tower',
                           'vectorization', 'language_modeling',
                           'sequence_labeling', 'data_augmentation',
                           'sequence_generation', 'geep_classify',
                           "text2image_generation", 'clip'
                       ],
                       help='name of the application')

    group.add_argument('--distributed_backend',
                       default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')

    group.add_argument('--sequence_length',
                       type=int,
                       default=16,
                       help='Maximum sequence length to process.')

    group.add_argument(
        '--micro_batch_size',
        '--train_batch_size',
        type=int,
        default=2,
        help='Batch size per model instance (local batch size). '
        'Global batch size is local batch size times data '
        'parallel size times number of micro batches.')

    group.add_argument('--local_rank',
                       type=int,
                       default=None,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--checkpoint_dir',
                       '--checkpoint_path',
                       default=None,
                       type=str,
                       help='The model checkpoint dir.')

    group.add_argument('--modelzoo_base_dir',
                       default='',
                       type=str,
                       required=False,
                       help='The Base directories of modelzoo')

    group.add_argument('--do_lower_case',
                       action='store_true',
                       help='Set this flag if you are using an uncased model.')

    group.add_argument('--epoch_num',
                       default=3.0,
                       type=float,
                       help='Total number of training epochs to perform.')

    group.add_argument('--save_checkpoint_steps', type=int, default=None)

    group.add_argument('--save_all_checkpoints',
                       action='store_true',
                       help='Whether to save all checkpoints per eval steps.')

    group.add_argument('--learning_rate',
                       default=5e-5,
                       type=float,
                       help='The initial learning rate for Adam.')

    group.add_argument('--weight_decay',
                       '--wd',
                       default=1e-4,
                       type=float,
                       metavar='W',
                       help='weight decay')

    group.add_argument('--max_grad_norm',
                       '--mn',
                       default=1.0,
                       type=float,
                       help='Max grad norm')

    group.add_argument('--optimizer_type',
                       '--optimizer',
                       default='BertAdam',
                       type=str,
                       choices=[
                           'BertAdam', 'Adam',
                           'AdamW', 'SGD',
                       ],
                       help='name of the optimizer')

    group.add_argument(
        '--warmup_proportion',
        default=0.1,
        type=float,
        help=
        'Proportion of training to perform linear learning rate warmup for. '
        'E.g., 0.1 = 10%% of training.')
    group.add_argument('--logging_steps',
                       default=100,
                       type=int,
                       help='logging steps while training')
    group.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
    )

    group.add_argument('--resume_from_checkpoint',
                       type=str,
                       default=None,
                       help='Resume training process from checkpoint')

    group.add_argument('--export_tf_checkpoint_type',
                       type=str,
                       default='easytransfer',
                       choices=['easytransfer', 'google', 'none'],
                       help='Which type of checkpoint you want to export')

    group.add_argument('--input_schema',
                       type=str,
                       default=None,
                       help='Only for csv data, the schema of input table')
    group.add_argument('--first_sequence',
                       type=str,
                       default=None,
                       help='Which column is the first sequence mapping to')
    group.add_argument('--second_sequence',
                       type=str,
                       default=None,
                       help='Which column is the second sequence mapping to')
    group.add_argument('--label_name',
                       type=str,
                       default=None,
                       help='Which column is the label mapping to')
    group.add_argument('--label_enumerate_values',
                       type=str,
                       default=None,
                       help='Which column is the label mapping to')
    group.add_argument('--output_schema',
                       type=str,
                       default='',
                       help='The schema of the output results')
    group.add_argument('--append_cols',
                       type=str,
                       default=None,
                       help='The schema of the output results')

    group.add_argument('--predict_slice_size',
                       default=4096,
                       type=int,
                       help='Predict slice size')
    group.add_argument('--predict_queue_size',
                       default=1024,
                       type=int,
                       help='Predict queue size')
    group.add_argument('--predict_thread_num',
                       default=2,
                       type=int,
                       help='Predict Thread num')
    group.add_argument('--predict_table_read_thread_num',
                       default=16,
                       type=int,
                       help='Predict Table Read Thread Num')
    group.add_argument('--restore_works_dir',
                       default='./.easynlp_predict_restore_works_dir',
                       type=str,
                       help='(for PAI-TF fail-over)')
    group.add_argument('--ps_hosts',
                       default='',
                       type=str,
                       help='PS hosts (for PAI-TF)')
    group.add_argument('--chief_hosts',
                       default='',
                       type=str,
                       help='Chief hosts (for PAI-TF)')
    group.add_argument('--job_name',
                       default=None,
                       type=str,
                       help='Name of the job (for PAI-TF)')
    group.add_argument('--task_index',
                       default=0,
                       type=int,
                       help='Index of the task (for PAI-TF)')
    group.add_argument('--task_count',
                       default=1,
                       type=int,
                       help='Number of the task (for PAI-TF)')
    group.add_argument('--is_chief',
                       default='',
                       type=str,
                       help='is chief (for PAI-TF)')

    group.add_argument('--worker_count',
                       default=1,
                       type=int,
                       help='Count of workers/servers')
    group.add_argument('--worker_gpu',
                       default=-1,
                       type=int,
                       help='Count of GPUs in each worker')
    group.add_argument('--worker_cpu',
                       default=-1,
                       type=int,
                       help='Count of CPUs in each worker')
    group.add_argument('--master_port',
                       default=23456,
                       type=int,
                       help='Port of master node')
    group.add_argument('--worker_hosts',
                       default=None,
                       type=str,
                       help='Worker hosts (for PAI-TF)')

    group.add_argument('--use_amp',
                       action='store_true',
                       help='Enable amp, default value is False')
    group.add_argument('--use_torchacc',
                       action='store_true',
                       help='Enable torchacc, default value is False')
    group.add_argument('--data_threads',
                       default=10,
                       type=int,
                       help='Count of CPUs to process data')

    return parser
