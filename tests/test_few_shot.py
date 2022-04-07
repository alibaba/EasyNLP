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


class TestFewshotLearning(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName=methodName)
        self.model = 'bert-small-uncased'
        self.type = 'pet_fewshot'
        self.label_desc = '否,能'
        self.pattern = 'sent1,label,用,sent2,概括。'

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
            cwd.joinpath('fewshot_logs').mkdir(exist_ok=True)
            if not cwd.joinpath('fewshot_train.tsv').exists():
                cls.download_file(
                    'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fewshot_learning/fewshot_train.tsv'
                )
            if not cwd.joinpath('fewshot_dev.tsv').exists():
                cls.download_file(
                    'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/fewshot_learning/fewshot_dev.tsv'
                )

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     with pathlib.Path.cwd() as cwd:
    #         cwd.joinpath('few_shot_train.tsv').unlink()
    #         cwd.joinpath('few_shot_dev.tsv').unlink()
    #         rmtree(cwd.joinpath('few_shot_logs'))

    def test_0_train(self):
        argv = [
            '--app_name=text_match', '--mode=train', '--worker_count=1',
            '--worker_gpu=1', '--tables=./fewshot_train.tsv,./fewshot_dev.tsv',
            '--input_schema=sid:str:1,sent1:str:1,sent2:str:1,label:str:1',
            '--first_sequence=sent1', '--second_sequence=sent2',
            '--label_name=label', '--label_enumerate_values=0,1',
            '--checkpoint_dir=./fewshot_model/', '--learning_rate=1e-5',
            '--epoch_num=1', '--random_seed=42', '--save_checkpoint_steps=100',
            '--sequence_length=512', '--micro_batch_size=8',
            '--user_defined_parameters=pretrain_model_name_or_path={} type={} label_desc={} pattern={} enable_fewshot={}'
            .format(self.model, self.type, self.label_desc, self.pattern, True)
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('fewshot_logs/test0.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError:
            traceback.print_exc()
            self.fail('Test 0 failed.')

        with pathlib.Path('./fewshot_model/') as p:
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

    def test_1_predict(self):
        argv = [
            '--app_name=text_match', '--mode=predict',
            '--tables=./fewshot_train.tsv,./fewshot_dev.tsv',
            '--outputs=pred.tsv', '--output_schema=predictions',
            '--input_schema=sid:str:1,sent1:str:1,sent2:str:1,label:str:1',
            '--worker_count=1', '--worker_gpu=1', '--first_sequence=sent1',
            '--second_sequence=sent2', '--label_name=label',
            '--append_cols=sid,label', '--label_enumerate_values=0,1',
            '--checkpoint_dir=./fewshot_model/', '--micro_batch_size=8',
            '--sequence_length=512',
            '--user_defined_parameters=pretrain_model_name_or_path={} type={} label_desc={} pattern={} enable_fewshot={}'
            .format(self.model, self.type, self.label_desc, self.pattern, True)
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('fewshot_logs/test1.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError:
            traceback.print_exc()
            self.fail('Test 1 failed.')

        with pathlib.Path() as p:
            self.assertTrue((p / 'pred.tsv').exists())


if __name__ == '__main__':
    unittest.main()
