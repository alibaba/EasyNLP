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

import pathlib
import subprocess
import traceback
import unittest
from shutil import rmtree

import requests

EXEC_CMD = 'easynlp'


class TestClassificationDistillation(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName=methodName)

        self.teacher_model = 'bert-base-uncased'
        self.student_model = 'bert-small-uncased'

    @staticmethod
    def download_file(url):
        local_filename = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    @classmethod
    def setUpClass(cls) -> None:
        with pathlib.Path.cwd() as cwd:
            cwd.joinpath('logs').mkdir(exist_ok=True)
            if not cwd.joinpath('train.tsv').exists():
                cls.download_file(
                    'http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/train.tsv'
                )
            if not cwd.joinpath('dev.tsv').exists():
                cls.download_file(
                    'http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/classification/dev.tsv'
                )

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     with pathlib.Path.cwd() as cwd:
    #         cwd.joinpath('train.tsv').unlink()
    #         cwd.joinpath('dev.tsv').unlink()
    #         rmtree(cwd.joinpath('results'))

    def test_0_teacher_train(self):
        argv = [
            '--app_name=text_classify',
            '--mode=train',
            '--worker_count=1',
            '--worker_gpu=1',
            '--tables=train.tsv,dev.tsv',
            '--input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--label_name=label',
            '--label_enumerate_values=0,1',
            '--checkpoint_dir=./results/mrpc_teacher',
            '--learning_rate=3e-5',
            '--epoch_num=1',
            '--random_seed=42',
            '--save_checkpoint_steps=100',
            '--sequence_length=128',
            '--micro_batch_size=32',
            '--user_defined_parameters=pretrain_model_name_or_path={}'.format(
                self.teacher_model),
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test0.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError:
            traceback.print_exc()
            self.fail('Test 0 failed.')

        with pathlib.Path('results/mrpc_teacher') as p:
            self.assertTrue(p.exists())
            expected_files = [
                'config.json',
                'label_mapping.json',
                'pytorch_model.bin',
                'pytorch_model.meta.bin',
                'train_config.json',
                'vocab.txt',
            ]
            files = [f.name for f in p.iterdir()]
            for exp in expected_files:
                self.assertTrue(exp in files)

    def test_1_save_logits(self):
        argv = [
            '--mode=predict',
            '--worker_count=1',
            '--worker_gpu=1',
            '--tables=train.tsv',
            '--outputs=pred.tsv',
            '--input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--output_schema=logits',
            '--checkpoint_path=./results/mrpc_teacher',
            '--micro_batch_size=32',
            '--sequence_length=128',
            '--app_name=text_classify',
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test1.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError:
            traceback.print_exc()
            self.fail('Test 1 failed.')

        with pathlib.Path() as p:
            self.assertTrue((p / 'pred.tsv').exists())

    def test_2_student_finetune(self):
        user_defined_parameters = {
            'pretrain_model_name_or_path': self.student_model,
            'enable_distillation': True,
            'type': 'vanilla_kd',
            'logits_name': 'logits',
            'logits_saved_path': 'pred.tsv',
            'temperature': 10,
            'alpha': 0.25,
        }

        kv_udp = ' '.join(
            [f'{k}={v}' for k, v in user_defined_parameters.items()])

        argv = [
            '--app_name=text_classify',
            '--mode=train',
            '--worker_count=1',
            '--worker_gpu=1',
            '--tables=train.tsv,dev.tsv',
            '--input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1,logits:float:2',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--label_name=label',
            '--label_enumerate_values=0,1',
            '--checkpoint_dir=./results/mrpc_student',
            '--learning_rate=3e-5',
            '--epoch_num=1',
            '--random_seed=42',
            '--save_checkpoint_steps=200',
            '--sequence_length=128',
            '--micro_batch_size=32',
            '--user_defined_parameters=' + kv_udp,
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test2.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError:
            traceback.print_exc()
            self.fail('Test 2 failed.')

        with pathlib.Path('results/mrpc_student') as p:
            self.assertTrue(p.exists())
            expected_files = [
                'config.json',
                'label_mapping.json',
                'pytorch_model.bin',
                'pytorch_model.meta.bin',
                'train_config.json',
                'vocab.txt',
            ]
            files = [f.name for f in p.iterdir()]
            for exp in expected_files:
                self.assertTrue(exp in files)


if __name__ == '__main__':
    unittest.main()
