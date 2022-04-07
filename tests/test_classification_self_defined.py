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


class TestClassificationSelfDefined(unittest.TestCase):
    def test_0_train(self):
        argvs = "python test_classification_main.py   \
                 --mode=train --tables=train.tsv,dev.tsv   \
                 --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 --first_sequence=sent1   \
                 --second_sequence=sent2 --label_name=label --label_enumerate_values=0,1   \
                 --checkpoint_dir=./classification_model/   \
                 --learning_rate=3e-5 --epoch_num=1 --random_seed=42  --save_checkpoint_steps=100   \
                 --sequence_length=128 --micro_batch_size=32 --app_name=text_classify   \
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


if __name__ == '__main__':
    unittest.main()
