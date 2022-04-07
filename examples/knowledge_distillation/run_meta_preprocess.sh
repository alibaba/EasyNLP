# Prework
set -x
set -e

curl=/workspace/project_all/EasyNLP/examples/knowledge_distillation/metakd
cd ${curl}

# 1. Download Amazon dataset
if [ ! -d data ];then
  mkdir data
fi

cd data
if [ ! -f ./SENTI/dev.tsv ];then
  # wget https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/datasets/domain_data/domain_sentiment_data.tar.gz
  tar -zxvf domain_sentiment_data.tar.gz
fi
cd ..

# 2. Split the dataset
if [ ! -f data/SENTI/dev.tsv ];then
  python generate_senti_data.py
fi

# 3. extract bert embedding
if [ ! -f data/SENTI/train.embeddings.tsv ];then
  python extract_embeddings.py \
    --bert_path ~/.easynlp/modelzoo/bert-base-uncased \
    --input data/SENTI/train.tsv \
    --output data/SENTI/train.embeddings.tsv \
    --task_name senti --gpu 7
fi

# 4. generate instance weights
if [ ! -f data/SENTI/train_with_weights.tsv ];then
  python generate_meta_weights.py \
    data/SENTI/train.embeddings.tsv \
    data/SENTI/train_with_weights.tsv \
    books,dvd,electronics,kitchen
fi

# 5. generate dev file
if [ ! -f data/SENTI/dev_standard.tsv ];then
  python generate_dev_file.py \
    --input data/SENTI/dev.tsv \
    --output data/SENTI/dev_standard.tsv
fi