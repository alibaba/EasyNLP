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


class TestLabeling(unittest.TestCase):
    def test_0_train(self):
        argvs = "easynlp  \
                  --mode=train \
                  --worker_gpu=1 \
                  --tables=train.csv,dev.csv \
                  --input_schema=content:str:1,label:str:1 \
                  --first_sequence=content \
                  --label_name=label \
                  --label_enumerate_values=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
                  --app_name=sequence_labeling \
                  --checkpoint_dir=./labeling_model \
                  --learning_rate=1e-4  \
                  --epoch_num=1  \
                  --logging_steps=100 \
                  --save_checkpoint_steps=100 \
                  --sequence_length=128 \
                  --micro_batch_size=64 \
                  --user_defined_parameters='pretrain_model_name_or_path=bert-small-uncased' \
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

        self.assertTrue('./labeling_model/pytorch_model.bin')
        self.assertTrue('./labeling_model/pytorch_model.meta.bin')
        self.assertTrue('./labeling_model/model.ckpt.data-00000-of-00001')
        self.assertTrue('./labeling_model/model.ckpt.index')
        self.assertTrue('./labeling_model/model.ckpt.meta')
        self.assertTrue('./labeling_model/config.json')
        self.assertTrue('./labeling_model/vocab.txt')
        self.assertTrue('./labeling_model/label_mapping.json')

    def test_1_evaluate(self):
        argvs = 'easynlp  \
                  --mode=evaluate \
                  --worker_gpu=1 \
                  --tables=dev.csv \
                  --input_schema=content:str:1,label:str:1 \
                  --first_sequence=content \
                  --label_name=label \
                  --label_enumerate_values=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
                  --app_name=sequence_labeling \
                  --checkpoint_path=./labeling_model \
                  --sequence_length=128 \
                  --micro_batch_size=64 \
                 '

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
        argvs = 'easynlp \
                  --mode=predict \
                  --worker_gpu=1 \
                  --app_name=sequence_labeling \
                  --tables=test.csv \
                  --outputs=test.pred.csv \
                  --input_schema=content:str:1,label:str:1 \
                  --first_sequence=content \
                  --sequence_length=128 \
                  --output_schema=predictions,output \
                  --append_cols=label \
                  --checkpoint_path=./labeling_model \
                  --micro_batch_size 32 \
                 '

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


if __name__ == '__main__':
    unittest.main()
