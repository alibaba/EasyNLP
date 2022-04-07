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


class TestClassification(unittest.TestCase):
    def test_predict(self):
        argvs = 'easynlp \
                    --mode=predict \
                    --worker_gpu=1 \
                    --tables=dev2.tsv \
                    --outputs=dev.pred.tsv \
                    --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
                    --output_schema=pooler_output,first_token_output,all_hidden_outputs \
                    --first_sequence=sent1 \
                    --append_cols=label \
                    --checkpoint_dir=bert-small-uncased \
                    --micro_batch_size=32 \
                    --sequence_length=128 \
                    --app_name=vectorization \
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

        with open('./dev.pred.tsv') as f:
            for line in f:
                pooler_output, first_token_output, all_hidden_output, label = line.strip(
                ).split('\t')
                pooler_output = pooler_output.split(',')
                first_token_output = first_token_output.split(',')
                assert len(
                    pooler_output
                ) == 512, 'len of pooler_output should be 512 (bert-small)'
                assert len(
                    first_token_output
                ) == 512, 'len of first_token_output should be 512 (bert-small)'


if __name__ == '__main__':
    unittest.main()
