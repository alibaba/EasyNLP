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
"""EasyNLP initialization."""
import os
import random
import time

import numpy as np
import torch

from .global_vars import (get_args, get_tensorboard_writer,
                          parse_user_defined_parameters, set_global_variables)
from .io_utils import OSSIO, TFOSSIO, io, parse_oss_buckets
from .logger import init_logger


def initialize_easynlp(extra_args_provider=None,
                       args_defaults={},
                       ignore_unknown_args=False):

    # Support both CPU and GPU train/eval/predict
    set_global_variables(extra_args_provider=extra_args_provider,
                         args_defaults=args_defaults,
                         ignore_unknown_args=ignore_unknown_args)
    args = get_args()

    _initialize_distributed()

    # Random seeds for reproducibility.
    if args.rank == 0:
        print('> setting random seeds to {} ...'.format(args.random_seed))
    _set_random_seed(args.random_seed)

    #this env is for predictor
    #os.environ['TF_FAILOVER_RESTORE_WORKS_DIR'] = args.restore_works_dir
    os.environ['EASYNLP_MODELZOO_BASE_DIR'] = args.modelzoo_base_dir
    os.environ['EASYNLP_IS_MASTER'] = str(args.is_master_node)
    os.environ['EASYNLP_N_GPUS'] = str(args.n_gpu)

    init_logger(local_rank=args.rank)
    if args.buckets is not None:
        init_oss_io(args)
    if args.mode == 'train' or not args.checkpoint_dir:
        from . import get_pretrain_model_path
        args.pretrained_model_name_or_path = parse_user_defined_parameters(
            args.user_defined_parameters).get('pretrain_model_name_or_path',
                                              None)
        args.pretrained_model_name_or_path = get_pretrain_model_path(
            args.pretrained_model_name_or_path)
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    args.data_threads = max(args.data_threads, 5)
    # Compile dependencies.
    #_compile_dependencies()

    # No continuation function
    return args


def _compile_dependencies():

    args = get_args()

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(time.time() -
                                                        start_time),
              flush=True)


def _initialize_distributed():
    """Initialize torch.distributed and mpu."""
    args = get_args()

    device_count = torch.cuda.device_count()

    if device_count > 0:
        args.distributed_backend = 'nccl'
    else:
        args.distributed_backend = 'gloo'

    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                'torch distributed is already initialized, '
                'skipping initialization ...',
                flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
        # Call the init process
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(backend=args.distributed_backend,
                                             world_size=args.world_size,
                                             rank=args.rank,
                                             init_method=init_method)

        torch.distributed.barrier()
        print('Init dist done. World size: {}, rank {}, l_rank {}'.format(
            args.world_size, args.rank, args.local_rank))


def _set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #else:
    #    raise ValueError(
    #        'Seed ({}) should be a positive integer.'.format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg,
                            str(getattr(args, arg)),
                            global_step=args.iteration)


def init_oss_io(cfg):
    if 'role_arn' in cfg.buckets:
        new_io = TFOSSIO()
    else:
        access_key_id, access_key_secret, hosts, buckets = parse_oss_buckets(
            cfg.buckets)
        if cfg.modelzoo_base_dir and 'oss://' in cfg.modelzoo_base_dir:
            _, _, mz_hosts, mz_buckets = parse_oss_buckets(
                cfg.modelzoo_base_dir)
            hosts += mz_hosts
            buckets += mz_buckets
        new_io = OSSIO(access_key_id=access_key_id,
                       access_key_secret=access_key_secret,
                       hosts=hosts,
                       buckets=buckets)
    io.set_io(new_io)


def init_odps_io(odps_config):
    os.environ['ODPS_CONFIG_FILE_PATH'] = odps_config
