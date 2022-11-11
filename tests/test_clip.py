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

import subprocess
import unittest

from sklearn.metrics import accuracy_score, f1_score


class TestCLIP(unittest.TestCase):
    def test_0_train(self):
        argvs = "easynlp \
                --mode train \
                --worker_gpu=1 \
                --tables=./MUGE_MR_train_base64_part.tsv,./MUGE_MR_valid_base64_part.tsv \
                --input_schema=text:str:1,image:str:1 \
                --first_sequence=text \
                --second_sequence=image \
                --checkpoint_dir=./clip_cn_model/ \
                --learning_rate=1e-6  \
                --epoch_num=1  \
                --random_seed=42 \
                --logging_steps=100 \
                --save_checkpoint_steps 200 \
                --sequence_length=32 \
                --micro_batch_size=32 \
                --app_name=clip \
                --save_all_checkpoints \
                --user_defined_parameters='pretrain_model_name_or_path=alibaba-pai/clip_chinese_roberta_base_vit_base'  \
                "

        print(argvs)
        try:
            p = subprocess.Popen(argvs,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True)
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                if 'returned non-zero exit status 1' in line.rstrip().decode(
                        'utf-8'):
                    p.stdout.close()
                    raise RuntimeError
                print(line.rstrip().decode('utf-8'))
            p.stdout.close()
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError

        self.assertTrue('./clip_model/pytorch_model.bin')
        self.assertTrue('./clip_model/config.json')

    def test_1_evaluate(self):
        argvs = "easynlp \
                --mode evaluate \
                --worker_gpu=1 \
                --tables=./MUGE_MR_valid_base64_part.tsv \
                --input_schema=text:str:1,image:str:1 \
                --first_sequence=text \
                --second_sequence=image \
                --checkpoint_dir=./clip_cn_model \
                --random_seed=42 \
                --logging_steps=100 \
                --save_checkpoint_steps=500 \
                --sequence_length=32 \
                --micro_batch_size=32 \
                --app_name=clip \
                 "

        print(argvs)
        try:
            p = subprocess.Popen(argvs,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True)
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                if 'returned non-zero exit status 1' in line.rstrip().decode(
                        'utf-8'):
                    p.stdout.close()
                    raise RuntimeError
                print(line.rstrip().decode('utf-8'))
            p.stdout.close()
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError

    def test_2_predict(self):
        argvs = "easynlp \
                --mode predict \
                --worker_gpu=1 \
                --tables=./MUGE_MR_test_base64_part_text.tsv \
                --input_schema=text:str:1 \
                --output_schema=text_feat \
                --outputs ./text_feat.tsv \
                --first_sequence=text \
                --checkpoint_dir=./clip_cn_model/ \
                --random_seed=42 \
                --logging_steps=100 \
                --save_checkpoint_steps=500 \
                --sequence_length=32 \
                --micro_batch_size=2 \
                --app_name=clip \
                 "

        print(argvs)
        try:
            p = subprocess.Popen(argvs,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True)
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                if 'returned non-zero exit status 1' in line.rstrip().decode(
                        'utf-8'):
                    p.stdout.close()
                    raise RuntimeError
                print(line.rstrip().decode('utf-8'))
            p.stdout.close()
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError
        self.assertTrue('./text_feat.tsv')


if __name__ == '__main__':
    test = TestCLIP()
    test.test_0_train()
    test.test_1_evaluate()
    test.test_2_predict()
