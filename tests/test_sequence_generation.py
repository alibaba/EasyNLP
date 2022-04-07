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
from collections import defaultdict

from rouge import Rouge


class TestGeneration(unittest.TestCase):
    def test_0_train(self):
        argvs = "easynlp \
                --app_name=sequence_generation \
                --mode=train \
                --worker_gpu=1 \
                --tables=./cn_train.tsv,./cn_dev.tsv  \
                --input_schema=title_tokens:str:1,content_tokens:str:1 \
                --first_sequence=content_tokens \
                --second_sequence=title_tokens \
                --label_name=title_tokens \
                --checkpoint_dir=./sequence_generation_model \
                --micro_batch_size=4 \
                --sequence_length=512 \
                --epoch_num 1\
                --learning_rate 1e-5 \
                --save_checkpoint_steps=150 \
                --export_tf_checkpoint_type none \
                --user_defined_parameters 'pretrain_model_name_or_path=alibaba-pai/mt5-title-generation-zh copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5' \
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

        self.assertTrue('./sequence_generation_model/pytorch_model.bin')
        self.assertTrue('./sequence_generation_model/pytorch_model.meta.bin')
        self.assertTrue('./sequence_generation_model/train_config.json')
        self.assertTrue('./sequence_generation_model/config.json')
        self.assertTrue('./sequence_generation_model/vocab.txt')
        self.assertTrue('./sequence_generation_model/label_mapping.json')

    def test_1_evaluate(self):
        argvs = "easynlp \
                --app_name=sequence_generation \
                --mode=evaluate \
                --worker_gpu=1 \
                --tables=./cn_dev.tsv  \
                --input_schema=title:str:1,content:str:1,title_tokens:str:1,content_tokens:str:1,tag:str:1 \
                --output_schema=predictions,beams \
                --append_cols=title_tokens,content,tag \
                --first_sequence=content_tokens \
                --second_sequence=title_tokens \
                --checkpoint_dir=./sequence_generation_model \
                --micro_batch_size=32 \
                --sequence_length 512 \
                --user_defined_parameters 'copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5' \
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
                --app_name=sequence_generation \
                --mode predict \
                --worker_gpu=1 \
                --tables=./cn_dev.tsv  \
                --outputs=./cn.preds.txt \
                --input_schema=title:str:1,content:str:1,title_tokens:str:1,content_tokens:str:1,tag:str:1 \
                --output_schema=predictions,beams \
                --append_cols=title_tokens,content,tag \
                --first_sequence=content_tokens \
                --checkpoint_dir=./sequence_generation_model \
                --micro_batch_size=32 \
                --sequence_length 512 \
                --user_defined_parameters 'copy=false max_encoder_length=512 min_decoder_length=12 max_decoder_length=32 no_repeat_ngram_size=2 num_beams=5 num_return_sequences=5' \
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

        pred_titles = list()
        grt_titles = list()
        pred_dict = defaultdict(list)
        grt_dict = defaultdict(list)
        with open('./cn.preds.txt') as f:
            for line in f:
                pred, candi, grt, _, tag = line.strip().split('\t')
                pred = ' '.join(pred.replace(' ', ''))
                grt = ' '.join(grt.replace(' ', ''))
                pred_titles.append(pred)
                grt_titles.append(grt)
                pred_dict[tag].append(pred)
                grt_dict[tag].append(grt)

        print('=' * 10 + ' Overall ' + '=' * 10)
        rouge = Rouge()
        scores = rouge.get_scores(pred_titles, grt_titles, avg=True)
        print('Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f}'.format(
            scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100,
            scores['rouge-l']['f'] * 100))

        for tag in pred_dict:
            print('=' * 10 + ' %s ' % tag + '=' * 10)
            scores = rouge.get_scores(pred_dict[tag], grt_dict[tag], avg=True)
            print('Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f}'.format(
                scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100,
                scores['rouge-l']['f'] * 100))
        self.assertTrue(scores['rouge-1']['f'] > 0.4)


if __name__ == '__main__':
    unittest.main()
