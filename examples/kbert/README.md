

# K-BERT

This cod is the re-implementation of K-BERT: Enabling Language Representation with Knowledge Graph.

K-BERT is a knowledge-enabled language representation model with knowledge graphs (KGs), in which triples are injected into the sentences as domain knowledge. K-BERT introduces soft- position and visible matrix to limit the impact of knowledge. It can easily inject domain knowledge into the models by equipped with a KG without pre-training by-self because it is capable of loading model parameters from the pre- trained BERT. 

### Quick start

This is a tutorial of a quick start of sentence classification and NER task of K-BERT.

Firstly, switch to the directory of kbert. 

```
cd examples/kbert
```

You can easily start the script of sentence classification by the following:

```
sh run_cls.sh
```

And you can start a NER task:

```
sh run_ner.sh
```

### Get data

```
if [ ! -f ./tmp/kbert_data ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/K-BERT/kbert_data.zip
  unzip kbert_data.zip
  rm -rf kbert_data.zip
  mkdir tmp/
  mv kbert_data tmp/
fi
```

### Test the code

```
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
python -m torch.distributed.launch $DISTRIBUTED_ARGS $basepath/kbert_cls.py \
  --mode train \
  --tables tmp/kbert_data/chnsenticorp/train.tsv,tmp/kbert_data/chnsenticorp/dev.tsv \
  --input_schema label:str:1,sid1:str:1,sent1:str:1 \
  --first_sequence sent1 \
  --label_name label\
  --label_enumerate_values 0,1 \
  --checkpoint_dir ./tmp/kbert_classification_model/ \
  --learning_rate 2e-5 \
  --epoch_num 2 \
  --random_seed 42 \
  --logging_steps 1 \
  --save_checkpoint_steps 50 \
  --sequence_length 128 \
  --micro_batch_size 10 \
  --app_name text_classify \
  --use_amp \
  --user_defined_parameters "pretrain_model_name_or_path=kbert-base-chinese kg_file=tmp/kbert_data/kbert_kgs/HowNet.spo"
```



### Citation

```
@inproceedings{DBLP:conf/aaai/LiuZ0WJD020,
  author    = {Weijie Liu and
               Peng Zhou and
               Zhe Zhao and
               Zhiruo Wang and
               Qi Ju and
               Haotang Deng and
               Ping Wang},
  title     = {{K-BERT:} Enabling Language Representation with Knowledge Graph},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {2901--2908},
  publisher = {{AAAI} Press},
  year      = {2020},
  url       = {https://ojs.aaai.org/index.php/AAAI/article/view/5681},
  timestamp = {Mon, 07 Mar 2022 16:58:18 +0100},
  biburl    = {https://dblp.org/rec/conf/aaai/LiuZ0WJD020.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
