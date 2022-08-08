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

import time
from abc import ABC, abstractmethod

import torch

from .arguments import parse_args, parse_args_for_cli

_GLOBAL_ARGS = None
_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None
_GLOBAL_APP_PARAMETER_NAMES = {
    'embedding_size': 'int',
    'loss_type': 'str',
    'margin': 'float',
    'gamma': 'int',
    'two_tower': 'bool',
    'siamese': 'bool',
    'multi_label': 'bool',

    # Knowledge Distillation
    'enable_distillation': 'bool',
    'type': 'str',
    'logits_name': 'str',
    'logits_saved_path': 'str',
    'temperature': 'float',
    'alpha': 'float',

    # Data Augmentation
    'expansion_rate': 'int',
    'mask_proportion': 'float',
    'remove_blanks': 'bool',
    'append_original': 'bool',

    # Fewshot Learning
    'enable_fewshot': 'bool',
    'type': 'str',
    'label_desc': 'str',
    'pattern': 'str', 

    # Image caption
    'enable_vit': 'bool',
    'enable_vqgan': 'bool', 

}
_GLOBAL_MODEL_PARAMETER_NAMES = ['vocab_size', 'hidden_size']


class NumMicroBatchesCalculator(ABC):
    def __init__(self):
        self.num_micro_batches = None
        self.current_global_batch_size = None

    def get(self):
        return self.num_micro_batches

    def get_current_global_batch_size(self):
        return self.current_global_batch_size

    @abstractmethod
    def update(self, consumed_samples, consistency_check):
        pass


class ConstantNumMicroBatches(NumMicroBatchesCalculator):
    def __init__(self, global_batch_size, micro_batch_size,
                 data_parallel_size):
        micro_batch_times_data_parallel = micro_batch_size * \
                                          data_parallel_size
        assert global_batch_size % micro_batch_times_data_parallel == 0, \
            'global batch size ({}) is not divisible by micro batch size ({})' \
            ' times data parallel size ({})'.format(global_batch_size,
                                                    micro_batch_size,
                                                    data_parallel_size)
        self.num_micro_batches = global_batch_size // \
                                 micro_batch_times_data_parallel
        assert self.num_micro_batches >= 1
        self.current_global_batch_size = global_batch_size

    def update(self, consumed_samples, consistency_check):
        pass


def build_num_microbatches_calculator(args):

    # Constant num micro-batches.
    num_microbatches_calculator = ConstantNumMicroBatches(
        args.global_batch_size, args.micro_batch_size, args.data_parallel_size)
    if args.rank == 0:
        print('setting number of micro-batches to constant {}'.format(
            num_microbatches_calculator.get()),
              flush=True)
    return num_microbatches_calculator


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples,
                                               consistency_check)


def get_tensorboard_writer():
    """Return tensorboard writer.

    It can be None so no need
    to check if it is initialized.
    """
    return _GLOBAL_TENSORBOARD_WRITER


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS


def set_global_variables(extra_args_provider=None,
                         args_defaults={},
                         ignore_unknown_args=False):
    """Set args, tensorboard-writer, and timers."""
    args = _parse_args(extra_args_provider=extra_args_provider,
                       defaults=args_defaults,
                       ignore_unknown_args=ignore_unknown_args)
    _set_tensorboard_writer(args)
    _set_timers()


def set_variables_for_cli(extra_args_provider=None,
                          defaults={},
                          ignore_unknown_args=False):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')

    _GLOBAL_ARGS = parse_args_for_cli(extra_args_provider=extra_args_provider,
                                      defaults=defaults,
                                      ignore_unknown_args=ignore_unknown_args)

    return _GLOBAL_ARGS


def parse_user_defined_parameters(user_defined_parameters):
    global _GLOBAL_APP_PARAMETER_NAMES
    global _GLOBAL_MODEL_PARAMETER_NAMES
    ret = {}
    app_parameters = {}
    model_parameters = {}
    if user_defined_parameters is not None:
        for ele in user_defined_parameters.split():
            key = ele.split('=')[0]
            value = ele.split('=')[1]
            if key in _GLOBAL_APP_PARAMETER_NAMES:
                value_type = _GLOBAL_APP_PARAMETER_NAMES[key]
                if value_type == 'int':
                    app_parameters[key] = int(value)
                elif value_type == 'float':
                    app_parameters[key] = float(value)
                elif value_type == 'bool':
                    if value == 'True':
                        app_parameters[key] = True
                    else:
                        app_parameters[key] = False
                else:
                    app_parameters[key] = value
            elif key in _GLOBAL_MODEL_PARAMETER_NAMES:
                model_parameters[key] = value
            else:
                ret[key] = value
    ret['app_parameters'] = app_parameters
    if len(model_parameters) != 0:
        ret['model_parameters'] = model_parameters
    return ret


def _parse_args(extra_args_provider=None,
                defaults={},
                ignore_unknown_args=False):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = parse_args(extra_args_provider=extra_args_provider,
                              defaults=defaults,
                              ignore_unknown_args=ignore_unknown_args)

    return _GLOBAL_ARGS


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir,
                max_queue=args.tensorboard_queue_size)
        except ModuleNotFoundError:
            print(
                'WARNING: TensorBoard writing requested but is not '
                'available (are you using PyTorch 1.1.0 or later?), '
                'no TensorBoard logs will be written.',
                flush=True)


def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


class _Timer:
    """Timer."""
    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""
    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer."""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                    torch.distributed.get_world_size() - 1):
                print(string, flush=True)
        else:
            print(string, flush=True)
