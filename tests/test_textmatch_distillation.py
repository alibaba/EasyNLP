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
import unittest

from sklearn.metrics import accuracy_score, f1_score

EXEC_CMD = 'easynlp'


class TestTextMatchDistillation(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName=methodName)
        pathlib.Path('logs').mkdir(exist_ok=True)

    def test_0_teacher_train(self):
        argv = [
            '--app_name=text_match',
            '--mode=train',
            '--worker_count=1',
            '--worker_gpu=1',
            '--tables=glue_data/MRPC/msr_paraphrase_train.txt,glue_data/MRPC/msr_paraphrase_test.txt',
            '--skip_first_line',
            '--input_schema=label:str:1,sid1:int:1,sid2:int:1,sent1:str:1,sent2:str:1',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--label_name=label',
            '--label_enumerate_values=0,1',
            '--pretrained_model_name_or_path=./../pretrained/bert-large-uncased',
            '--checkpoint_dir=./results/large_mrpc_teacher',
            '--learning_rate=3e-5',
            '--epoch_num=1',
            '--random_seed=42',
            '--save_checkpoint_steps=100',
            '--sequence_length=128',
            '--micro_batch_size=32',
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test0.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError

        with pathlib.Path('./results/large_mrpc_teacher') as p:
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
            '--tables=glue_data/MRPC/msr_paraphrase_train.txt',
            '--skip_first_line',
            '--outputs=pred.tsv',
            '--input_schema=label:str:1,sid1:int:1,sid2:int:1,sent1:str:1,sent2:str:1',
            '--output_schema=logits',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--checkpoint_path=./results/large_mrpc_teacher',
            '--micro_batch_size=32',
            '--sequence_length=128',
            '--app_name=text_match',
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test1.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError

        with pathlib.Path() as p:
            self.assertTrue((p / 'pred.tsv').exists())

    def test_2_student_finetune(self):
        user_defined_parameters = {
            'app_parameters': {
                'type': 'vanilla_kd',
                'logits_name': 'logits',
                'logits_saved_path': 'pred.tsv',
                'temperature': 10,
                'alpha': 0.25,
            }
        }
        argv = [
            '--app_name=text_match',
            '--mode=train',
            '--worker_count=1',
            '--worker_gpu=1',
            '--tables=glue_data/MRPC/msr_paraphrase_train.txt,glue_data/MRPC/msr_paraphrase_test.txt',
            '--skip_first_line',
            '--input_schema=label:str:1,sid1:int:1,sid2:int:1,sent1:str:1,sent2:str:1,logits:float:2',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--label_name=label',
            '--label_enumerate_values=0,1',
            '--pretrained_model_name_or_path=./../pretrained/bert_uncased_L-4_H-512_A-8',
            '--checkpoint_dir=./results/small_mrpc_student',
            '--learning_rate=3e-5',
            '--epoch_num=2',
            '--random_seed=42',
            '--save_checkpoint_steps=200',
            '--sequence_length=128',
            '--micro_batch_size=32',
            '--enable_distillation',
            '--user_defined_parameters={}'.format(user_defined_parameters),
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test2.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError

        with pathlib.Path('./results/small_mrpc_student') as p:
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

    def test_3_student_predict(self):
        argv = [
            '--mode=predict',
            '--worker_gpu=1',
            '--worker_count=1',
            '--tables=glue_data/MRPC/msr_paraphrase_test.txt',
            '--skip_first_line',
            '--outputs=student_pred.tsv',
            '--input_schema=label:str:1,sid1:int:1,sid2:int:1,sent1:str:1,sent2:str:1',
            '--output_schema=predictions,probabilities,logits,output',
            '--first_sequence=sent1',
            '--second_sequence=sent2',
            '--checkpoint_path=./results/small_mrpc_student',
            '--micro_batch_size=32',
            '--sequence_length=128',
            '--app_name=text_match',
        ]
        argv.insert(0, EXEC_CMD)
        print(' '.join(argv))

        try:
            with open('logs/test3.out', 'w+') as f:
                proc = subprocess.run(argv,
                                      stderr=subprocess.STDOUT,
                                      stdout=f,
                                      check=True)
        except subprocess.CalledProcessError as e:
            print(e.output.decode('utf-8'))
            raise RuntimeError

        with pathlib.Path() as p:
            self.assertTrue((p / 'student_pred.tsv').exists())

        y_preds = []
        y_trues = []
        with open('./student_pred.tsv') as f:
            for line in f:
                pred, *_, label = line.strip().split('\t')
                y_preds.append(int(pred))
                y_trues.append(int(label))
        print(f'Accuracy: {accuracy_score(y_trues, y_preds)}')
        print(f'F1: {f1_score(y_trues, y_preds)}')


if __name__ == '__main__':
    # unittest.main()
    ut = TestTextMatchDistillation()
    ut.test_2_student_finetune()
