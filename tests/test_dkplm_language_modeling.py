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


class TestClassification(unittest.TestCase):
    def test_0_train(self):
        argvs = "easynlp  \
                    --mode=train \
                    --worker_gpu=1 \
                    --tables=train_corpus.txt,dev_corpus.txt \
                    --learning_rate=1e-4  \
                    --epoch_num=1  \
                    --logging_steps=100 \
                    --save_checkpoint_steps=500 \
                    --sequence_length=128 \
                    --train_batch_size=16 \
                    --app_name=language_modeling \
                    --checkpoint_dir=./tmp \
                    --user_defined_parameters='pretrain_model_name_or_path=alibaba-pai/pai-dkplm-medical-base-zh entity_emb_file=entity_emb.txt rel_emb_file=rel_emb.txt' \
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
