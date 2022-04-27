# MeLL: Large-scale Extensible User Intent Classification for Dialogue Systems with Meta Lifelong Learning


## How to build the dataset
1. The dataset can be found in this [link](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/mell/data.tar.gz) 


## How to run the code
1. Preprocess the data
```bash
$ wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/mell/data.tar.gz
$ tar zxvf data.tar.gz
```

2. Run the initial learning stage of MeLL
```bash
$ sh run_mell_initial.sh
```

3. Run the lifelong learning stage of MeLL
```bash
$ sh run_mell_lifelong.sh
```


If you use this code, please cite the following paper. Thanks.

```
@inproceedings{kdd2021,
  author    = {Chengyu Wang and
                Haojie Pan and
                Yuan Liu and
                Kehan Chen and
                Minghui Qiu and
                Wei Zhou and
                Jun Huang and
                Haiqing Chen and
                Wei Lin and
                Deng Cai},
  title     = {MeLL: Large-scale Extensible User Intent Classification for Dialogue Systems with Meta Lifelong Learning},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2021},
  pages     = {3649–3659}
}
```
