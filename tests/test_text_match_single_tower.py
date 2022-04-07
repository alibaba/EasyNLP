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


class TestTextMatchSingleTower(unittest.TestCase):
    def test_0_train(self):
        argvs = "easynlp  \
                    --mode=train \
                    --worker_gpu=1 \
                    --tables=train.csv,dev.csv \
                    --input_schema=example_id:str:1,sent1:str:1,sent2:str:1,label:str:1,cate:str:1,score:str:1 \
                    --first_sequence=sent1 \
                    --second_sequence=sent2 \
                    --label_name=label \
                    --label_enumerate_values=0,1 \
                    --checkpoint_dir=./text_match_single_tower_model_dir \
                    --learning_rate=3e-5  \
                    --epoch_num=1  \
                    --random_seed=42 \
                    --save_checkpoint_steps=100 \
                    --sequence_length=128 \
                    --train_batch_size=32 \
                    --app_name=text_match \
                    --user_defined_parameters='pretrain_model_name_or_path=bert-small-uncased loss_type=hinge_loss margin=0.45 gamma=32'\
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

        self.assertTrue(
            './text_match_single_tower_model_dir/pytorch_model.bin')
        self.assertTrue(
            './text_match_single_tower_model_dir/pytorch_model.meta.bin')
        self.assertTrue('./text_match_single_tower_model_dir/config.json')
        self.assertTrue('./text_match_single_tower_model_dir/vocab.txt')
        self.assertTrue(
            './text_match_single_tower_model_dir/train_config.json')
        self.assertTrue(
            './text_match_single_tower_model_dir/label_mapping.json')


if __name__ == '__main__':
    unittest.main()
