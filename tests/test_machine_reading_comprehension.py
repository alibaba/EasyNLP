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


class TestMachineReadingComprehension(unittest.TestCase):
    def test_0_train(self):
        argvs = "easynlp  \
                  --mode=train \
                  --app_name=machine_reading_comprehension \
                  --worker_gpu=1 \
                  --tables=dev_squad.tsv,dev_squad.tsv \
                  --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
                  --first_sequence=question_text \
                  --second_sequence=context_text \
                  --sequence_length=384 \
                  --checkpoint_dir=./mrc_model \
                  --learning_rate=3.5e-5 \
                  --epoch_num=1 \
                  --random_seed=42 \
                  --save_checkpoint_steps=2000 \
                  --train_batch_size=16 \
                  --user_defined_parameters='pretrain_model_name_or_path=bert-base-uncased language=en answer_name=answer_text qas_id=qas_id start_position_name=start_position_character doc_stride=128 max_query_length=64' \
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

        self.assertTrue('./mrc_model/pytorch_model.bin')
        self.assertTrue('./mrc_model/pytorch_model.meta.bin')
        self.assertTrue('./mrc_model/model.ckpt.data-00000-of-00001')
        self.assertTrue('./mrc_model/model.ckpt.index')
        self.assertTrue('./mrc_model/model.ckpt.meta')
        self.assertTrue('./mrc_model/config.json')
        self.assertTrue('./mrc_model/vocab.txt')
        self.assertTrue('./mrc_model/label_mapping.json')

    def test_1_evaluate(self):
        argvs = "easynlp  \
                  --mode=evaluate \
                  --app_name=machine_reading_comprehension \
                  --worker_gpu=1 \
                  --tables=dev_squad.tsv \
                  --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
                  --first_sequence=question_text \
                  --second_sequence=context_text \
                  --sequence_length=384 \
                  --checkpoint_dir=./mrc_model \
                  --micro_batch_size=16 \
                  --user_defined_parameters='pretrain_model_name_or_path=bert-base-uncased language=en qas_id=qas_id answer_name=answer_text start_position_name=start_position_character doc_stride=128 max_query_length=64' \
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
                  --mode=predict \
                  --app_name=machine_reading_comprehension \
                  --worker_gpu=1 \
                  --tables=dev_squad.tsv \
                  --outputs=dev.pred.csv \
                  --input_schema=qas_id:str:1,context_text:str:1,question_text:str:1,answer_text:str:1,start_position_character:str:1,title:str:1 \
                  --output_schema=unique_id,best_answer,query,context \
                  --first_sequence=question_text \
                  --second_sequence=context_text \
                  --sequence_length=384 \
                  --checkpoint_dir=./mrc_model \
                  --micro_batch_size=256 \
                  --user_defined_parameters='pretrain_model_name_or_path=bert-base-uncased language=en qas_id=qas_id answer_name=answer_text start_position_name=start_position_character max_query_length=64 max_answer_length=30 doc_stride=128 n_best_size=10 output_answer_file=dev.ans.csv' \
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
