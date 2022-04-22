## EasyNLP for CLUE Benchmark
This code is implement for the benchmark **CLUE** (Chinese Language Understanding Evaluation Benchmark). 
More details can be found in 《[CLUE: A Chinese Language Understanding Evaluation Benchmark](https://aclanthology.org/2020.coling-main.419/).
The leaderboard is from [https://www.cluebenchmarks.com/rank.html](https://www.cluebenchmarks.com/rank.html).

In this project, we provide the baseline (i.e. bert-base-chinese) for evaluating each task. 
This project is very easy for you to train a model and submit your predicted results. Please have fun!

### Quick Start

#### Step1: Training Stage
You can choose one task, and run the script. For example, you can train the task csl on cuda device 0:
> bash run_clue.sh 0 train csl

#### Step2: Predicting Stage
After the training stage, you will obtain the trained model in default directory ``./tmp/benchmarks/clue/csl``. You can directly run the script for prediction over test set:
> bash run_clue.sh 0 test csl

Then, the result will be saved in ``./tmp/predict/clue/csl/test_prediction.json``. You can directly upload this file on the leaderboard.


### Statistics

The statistics of each task can be found in the follow:


| Task  | AFQMC  | CMNLI | CSL   | IFLYTEK | OCNLI  | TNEWS  | WSC  |
|-------|--------|----|-------|---|--------|--------|------|
| train | 34,334 | 391,782  | 20,000  | 12,133  | 50,000 | 53,360 | 1,244 |
| dev   | 4,316  | 12,426 | 3,000    | 2,599  | 3,000  | 10,000 | 304   |
| test  | 3,861  | 13,880 | 3,000   | 2,600 | 3,000  | 10,000 | 2,574  |


### Settings

We provide default hyper-parameters for each task, you can reset them by your self:


| Task            | AFQMC | CMNLI | CSL   | IFLYTEK | OCNLI  | TNEWS | WSC  |
|-----------------|-------|-------|-------|---------|--------|-------|------|
| learning rate   | 5e-5  | 3e-5  | 1e-5  | 5e-5    | 3e-5   | 5e-5  | 5e-5 |
| batch size      | 48    | 16    | 32    | 16      | 16     | 32    | 32   |
| sequence length | 256   | 128   | 256   | 128     | 128    | 128   | 128  |



### Results

We simply train for only 5 epoch for each task (50 epoch for WSC), the backbone we choose is bert-base-chinese
The results of dev set can be found in the follow:

| Task | AFQMC  | CMNLI  | CSL    | IFLYTEK | OCNLI  | TNEWS  | WSC    |
|------|--------|--------|--------|---------|--------|--------|--------|
| P    | 72.17% | 75.74% | 80.93% | 60.22%  | 78.31% | 57.52% | 63.49% |
| F1   | 52.96% | 75.74% | 81.71% | 60.22%  | 78.30% | 57.52% | 77.67% |

