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

import os
import subprocess
import sys
from socket import socket

from easynlp.utils import get_args, io
from easynlp.utils.global_vars import set_variables_for_cli
from easynlp.utils.initializer import init_oss_io

sys.path.append('./')
sys.path.append('../')


def main():
    os.environ['PYTHONUNBUFFERED'] = '1'
    # TODO: Need to modify here
    os.environ['PATH'] = '/opt/conda/envs/python3.6/bin:' + os.environ['PATH']
    set_variables_for_cli()
    args = get_args()
    if args.user_script is not None and args.user_entry_file is not None:
        # User self-defined code
        init_oss_io(args)
        print(io.__name__)
        assert 'oss://' in args.user_script
        io.download(args.user_script, os.path.basename(args.user_script))
        argvs = 'tar -zxvf {}'.format(os.path.basename(args.user_script))
        print(argvs)
        subprocess.check_output(argvs, stderr=subprocess.STDOUT, shell=True)
        assert args.user_entry_file.endswith('.py')
        cmd = [sys.executable, '-u']
        cmd.append('-m')
        cmd.append('torch.distributed.launch')
        cmd.append('--nproc_per_node')
        cmd.append(str(args.worker_gpu))
        cmd.append('--nnodes')
        cmd.append(str(args.worker_count))
        cmd.append('--node_rank')
        cmd.append(os.environ['RANK'])
        cmd.append('--master_addr')
        cmd.append(os.environ['MASTER_ADDR'])
        cmd.append('--master_port')
        cmd.append(os.environ['MASTER_PORT'])
        cmd.append(args.user_entry_file)
        cmd_str = ' '.join(cmd) + ' '
        cmd_str += args.user_defined_parameters + ' '
        cmd_str += '--buckets=' + "\"" + args.buckets + "\""
        cmd_str += ' --modelzoo_base_dir=' + "\"" + args.modelzoo_base_dir + "\""
        print(cmd_str)
        try:
            p = subprocess.Popen(cmd_str,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True)
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                print(line.rstrip().decode('utf-8'))
            p.stdout.close()
            returncode = p.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, p)
            p.stdout.close()
        except Exception as e:
            raise RuntimeError(e)
    else:
        with socket() as s:
            s.bind(('', 0))
            random_available_port = s.getsockname()[1]
        cmd = ['python']
        cmd.append('-m')
        cmd.append('torch.distributed.launch')
        cmd.append('--nproc_per_node')
        cmd.append(str(args.worker_gpu))
        cmd.append('--nnodes')
        cmd.append(str(args.worker_count))
        cmd.append('--node_rank')
        cmd.append('0')
        cmd.append('--master_addr')
        cmd.append('localhost')
        cmd.append('--master_port')
        cmd.append(str(random_available_port))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cmd.append(os.path.join(dir_path, 'appzoo/api.py'))
        cmd.append('--mode')
        cmd.append(args.mode)
        cmd.append('--tables')
        cmd.append(args.tables)
        if args.skip_first_line:
            cmd.append('--skip_first_line')
        if args.input_schema is not None:
            cmd.append('--input_schema')
            cmd.append(args.input_schema)
        if args.first_sequence is not None:
            cmd.append('--first_sequence')
            cmd.append(args.first_sequence)
        if args.second_sequence is not None:
            cmd.append('--second_sequence')
            cmd.append(args.second_sequence)

        if args.mode != 'predict':
            if args.label_name is not None:
                cmd.append('--label_name')
                cmd.append(args.label_name)
            if args.label_enumerate_values is not None:
                cmd.append('--label_enumerate_values')
                cmd.append(args.label_enumerate_values)

        cmd.append('--checkpoint_dir')
        cmd.append(args.checkpoint_dir)
        cmd.append('--export_tf_checkpoint_type')
        cmd.append(args.export_tf_checkpoint_type)
        cmd.append('--learning_rate')
        cmd.append(str(args.learning_rate))
        cmd.append('--epoch_num')
        cmd.append(str(args.epoch_num))
        cmd.append('--random_seed')
        cmd.append(str(args.random_seed))
        if args.mode == 'train':
            cmd.append('--save_checkpoint_steps')
            cmd.append(str(args.save_checkpoint_steps))
        if args.mode == 'predict':
            cmd.append('--predict_queue_size')
            cmd.append('1024')
            cmd.append('--predict_slice_size')
            cmd.append('4096')
            cmd.append('--predict_thread_num')
            cmd.append('1')
            cmd.append('--outputs')
            cmd.append(args.outputs)
            cmd.append('--output_schema')
            cmd.append(args.output_schema)
            cmd.append('--restore_works_dir')
            cmd.append(args.restore_works_dir)
            if args.append_cols is not None:
                cmd.append('--append_cols')
                cmd.append(args.append_cols)
        cmd.append('--sequence_length')
        cmd.append(str(args.sequence_length))
        cmd.append('--micro_batch_size')
        cmd.append(str(args.micro_batch_size))
        cmd.append('--app_name')
        cmd.append(args.app_name)
        if args.buckets is not None:
            cmd.append('--buckets')
            cmd.append(args.buckets)
        if args.modelzoo_base_dir != '':
            cmd.append('--modelzoo_base_dir')
            cmd.append(args.modelzoo_base_dir)
        if args.user_defined_parameters is not None:
            cmd.append('--user_defined_parameters')
            cmd.append(args.user_defined_parameters)

            # fewshot models need label data in predict mode!
            for ele in args.user_defined_parameters.split():
                key = ele.split('=')[0]
                value = ele.split('=')[1]
                if key == 'enable_fewshot' and value == 'True':
                    if args.mode == 'predict':
                        if args.label_name is not None:
                            cmd.append('--label_name')
                            cmd.append(args.label_name)
                        if args.label_enumerate_values is not None:
                            cmd.append('--label_enumerate_values')
                            cmd.append(args.label_enumerate_values)
                        break

        cmd_str = ' '.join(cmd)
        print(cmd_str)

        p = subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
