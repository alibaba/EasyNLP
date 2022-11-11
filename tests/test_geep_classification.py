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
    def test_0_train(self):
        argvs = "easynlp \
                --mode train \
                --worker_gpu=1 \
                --tables=train_toy.tsv,dev.tsv \
                --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
                --first_sequence=sent1 \
                --second_sequence=sent2 \
                --label_name=label \
                --label_enumerate_values=0,1 \
                --checkpoint_dir=./classification_model \
                --learning_rate=3e-5  \
                --epoch_num=1  \
                --random_seed=42 \
                --save_checkpoint_steps=50 \
                --sequence_length=128 \
                --micro_batch_size=4 \
                --app_name=geep_classify \
                --user_defined_parameters='geep_exit_num=8 pretrain_model_name_or_path=bert-base-uncased' \
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

        self.assertTrue('./classification_model/pytorch_model.bin')
        self.assertTrue('./classification_model/pytorch_model.meta.bin')
        self.assertTrue('./classification_model/config.json')
        self.assertTrue('./classification_model/vocab.txt')
        self.assertTrue('./classification_model/label_mapping.json')

    def test_1_evaluate(self):
        argvs = "easynlp \
                --mode=evaluate \
                --worker_gpu=1 \
                --tables=dev.tsv \
                --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
                --first_sequence=sent1 \
                --second_sequence=sent2 \
                --label_name=label \
                --label_enumerate_values=0,1 \
                --checkpoint_dir=./classification_model \
                --sequence_length=128 \
                --micro_batch_size=32 \
                --app_name=geep_classify \
                --user_defined_parameters='geep_threshold=0.1' \
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
                --worker_gpu=1 \
                --tables=dev.tsv \
                --outputs=dev.pred.tsv \
                --input_schema=label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1 \
                --output_schema=predictions,probabilities,logits,output \
                --append_cols=label \
                --first_sequence=sent1 \
                --second_sequence=sent2 \
                --checkpoint_path=./classification_model \
                --micro_batch_size=32 \
                --sequence_length=128 \
                --app_name=geep_classify \
                --user_defined_parameters='geep_threshold=0.1' \
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
        y_preds = list()
        y_trues = list()
        with open('./dev.pred.tsv') as f:
            for line in f:
                pred, *_, label = line.strip().split('\t')
                y_preds.append(int(pred))
                y_trues.append(int(label))
        print('Accuracy: ', accuracy_score(y_trues, y_preds))
        print('F1: ', f1_score(y_trues, y_preds))
        self.assertTrue(accuracy_score(y_trues, y_preds) > 0.1)
        self.assertTrue(f1_score(y_trues, y_preds) > 0.1)


if __name__ == '__main__':
    # unittest.main()
    test = TestClassification()
    test.test_0_train()
    test.test_1_evaluate()
    test.test_2_predict()
