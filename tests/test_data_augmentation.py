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


class TestDataAugmentation(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName=methodName)

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
        train_file = pathlib.Path('dev.tsv')
        if not train_file.exists():
            cls.download_file(
                'http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/sentence_classification/dev.tsv'
            )
        with open('dev.tsv', 'r') as rf, open('tmp/train_head.tsv', 'w') as wf:
            # Take the first 64 rows of data
            for _ in range(64):
                wf.write(rf.readline())

    @classmethod
    def tearDownClass(cls) -> None:
        with pathlib.Path.cwd() as cwd:
            for tsv in cwd.glob('*.tsv'):
                tsv.unlink()

    def test_0_data_augmentation(self):
        user_defined_parameters = {
            'pretrain_model_name_or_path': 'bert-small-uncased',
            'type': 'mlm_da',
            'expansion_rate': 2,
            'mask_proportion': 0.1,
            'remove_blanks': True,
            'append_original': True,
        }

        kv_udp = ' '.join(
            [f'{k}={v}' for k, v in user_defined_parameters.items()])

        argv = [
            '--app_name=data_augmentation',
            '--worker_count=1',
            '--worker_gpu=1',
            '--mode=predict',
            '--tables=tmp/train_head.tsv',
            '--input_schema=index:str:1,sent:str:1,label:str:1',
            '--first_sequence=sent',
            '--label_name=label',
            '--outputs=tmp/train_aug.tsv',
            '--output_schema=augmented_data',
            '--checkpoint_dir=_',
            '--sequence_length=128',
            '--micro_batch_size=8',
            '--user_defined_parameters=' + kv_udp,
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('tmp/da_test_0.out', 'w+') as f:
                subprocess.run(argv,
                               stderr=subprocess.STDOUT,
                               stdout=f,
                               check=True)
        except:
            traceback.print_exc()
            self.fail('Test 0 failed.')

        with pathlib.Path('tmp/train_aug.tsv') as f:
            self.assertTrue(f.exists())

        num_src_records = int(
            subprocess.getoutput(f'cat tmp/train_head.tsv | wc -l'))
        num_aug_records = int(
            subprocess.getoutput(f'cat tmp/train_aug.tsv | wc -l'))

        self.assertEqual(
            num_src_records * user_defined_parameters['expansion_rate'],
            num_aug_records,
        )


if __name__ == '__main__':
    unittest.main()
