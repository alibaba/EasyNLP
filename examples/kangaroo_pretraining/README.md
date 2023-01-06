

# Kangaroo

This code is the re-implementation of Kangaroo, a Unified Framework  for Learning Knowledge-Enhanced Language Representations in Closed Domains.

### Quick start

This is a tutorial of a quick start of model pre-training and downstream task of Kangaroo.

Firstly, switch to the directory of kbert. 

```
cd examples/kangaroo_pretrain
```

You can easily start the script of pretraining by the following:

```
sh run_pretrain.sh
```

And you can start a sentence classification task:

```
sh run_cls.sh
```

And you can start a NER task:

```
sh run_ner.sh
```



### Get data

```
if [ ! -f tmp/finData ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/finData.tar.gz
  tar -zxvf finData.tar.gz
  rm -rf finData.tar.gz
  mkdir tmp/
  mv finData tmp/
fi
```



### Data prepare

If you have the corpus and KG yourself and want to use our model, you can pre-process data as follows:

```
cd examples/kangaroo_pretrain
cd poincare_embedding
sh train-financial.sh

cd ..
python get_finData_embedding.py
python pos_neg_samples.py
```



### Test the code

```
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6009"
python -m torch.distributed.launch $DISTRIBUTED_ARGS $basepath/run_pretrain.py \
  --mode train \
  --tables tmp/finData/financial_corpus.txt \
  --checkpoint_dir tmp/kangaroo_pretrain_model \
  --learning_rate 2e-5 \
  --epoch_num 1 \
  --random_seed 42 \
  --logging_steps 1 \
  --save_checkpoint_steps 50 \
  --sequence_length 128 \
  --micro_batch_size 16 \
  --app_name language_modeling \
  --use_amp \
  --user_defined_parameters "pretrain_model_name_or_path=alibaba-pai/pai-kangaroo-base-chinese entity_file=tmp/finData/finEntity.csv rel_file=tmp/finData/finRelation.csv concept_emb_file=tmp/finData/finConceptEmbedding.npy samples_file=tmp/finData/finSamples4Level.npy"
```

