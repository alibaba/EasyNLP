#!/bin/bash
export NCCL_DEBUG=WARN
export LC_ALL=C.UTF-8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_DATASETS_OFFLINE=0
MASTER_ADDR=localhost
MASTER_PORT=6034
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

declare -A input_schema
input_schema['eprstmt']=sid,sent,label
input_schema['bustm']=sid,sent,sent2,label
input_schema['chid']=sid,sent2,sent,label
input_schema['csl']=sid,sent,sent2,label
input_schema['csldcp']=sent,label,sid
input_schema['iflytek']=label,label_desc,sent,sid
input_schema['ocnli']=sid,sent,sent2,label
input_schema['tnews']=label,label_desc,sent,sent2,sid
input_schema['cluewsc']=sid,label,sent

declare -A pattern
pattern['eprstmt']=label,满意,sent
pattern['bustm']=sent,和,sent2,意思,label,同
pattern['cluewsc']=下面句子的指代关系是否正确？,label,sent
pattern['csldcp']=sent,这句话表达的主题是,label
pattern['iflytek']=作为一款,label,应用，,sent
pattern['tnews']=以下新闻的主题是,label,。,sent,关键词：,sent2
pattern['ocnli']=sent,和,sent2,的关系是什么?,label
pattern['csl']=sent,关键词：,sent2,答案：,label

declare -A label_desc
label_desc['eprstmt']=很,不
label_desc['bustm']=不,相
label_desc['csldcp']=控制,民族,计算,动力,历史,土木,矿业,数学,地球,园艺,畜牧,电子,政治,材料,林业,航空,图书,中文,新闻,测绘,地质,社会,艺术,作物,船舶,物理,心理,农林,生物,口腔,环境,食品,医学,建筑,法学,水利,体育,卫生,力学,中医,经济,药学,教育,农工,水产,冶金,机械,兵器,纺织,植物,海洋,公管,化学,地理,光学,交通,石油,天文,核科,军事,农业,大气,电气,通信,工管,应经,哲学
label_desc['tnews']=故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,股票,农业,游戏
label_desc['cluewsc']=是,否
label_desc['iflytek']=打车,地图,网络,租车,同城,物流,婚庆,家政,交通,政务,社区,羊毛,魔幻,仙侠,卡牌,飞行,射击,休闲,动作,体育,棋牌,养成,策略,多人,辅助,约会,通讯,工作,论坛,婚恋,情侣,社交,生活,博客,新闻,漫画,小说,技术,教辅,问答,搞笑,杂志,百科,影视,求职,兼职,视频,短视,音乐,直播,电台,K歌,成人,小学,职考,公务,英语,视教,高教,成教,艺术,语言,旅游,综合,民航,铁路,酒店,行程,民宿,出国,工具,亲子,母婴,驾校,违章,汽车,交易,养车,行车,租房,买房,装修,电子,问诊,保健,医疗,减肥,美妆,菜谱,餐饮,体讯,运动,支付,保险,股票,借贷,理财,彩票,记账,银行,美颜,剪辑,修图,相机,绘画,二手,电商,团购,外卖,票务,超市,购物,笔记,办公,日程,女性,经营,收款,其他
label_desc['ocnli']=相关,无,相反
label_desc['csl']=错误,正确

declare -A label_name
label_name['eprstmt']=Positive,Negative
label_name['bustm']=0,1
label_name['chid']=0,1,2,3,4,5,6
label_name['csl']=0,1
label_name['csldcp']=控制科学与工程,民族学,计算机科学与技术,动力工程及工程热物理,历史学,土木工程,矿业工程,数学,地球物理学,园艺学,畜牧学/兽医学,电子科学与技术,政治学,材料科学与工程,林学/林业工程,航空宇航科学与技术,图书馆、情报与档案管理,中国语言文学,新闻传播学,测绘科学与技术,地质学/地质资源与地质工程,社会学,艺术学,作物学,船舶与海洋工程,物理学,心理学,农林经济管理,生物学/生物科学与工程,口腔医学,环境科学与工程,食品科学与工程,基础医学/临床医学,建筑学,法学,水利工程,体育学,公共卫生与预防医学,力学,中医学/中药学,理论经济学,药学,教育学,农业工程,水产,冶金工程,机械工程,兵器科学与技术,纺织科学与工程,植物保护,海洋科学,公共管理,化学/化学工程与技术,地理学,光学工程,交通运输工程,石油与天然气工程,天文学,核科学与技术,军事学,农业资源利用,大气科学,电气工程,信息与通信工程,工商管理,应用经济学,哲学
label_name['iflytek']=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118
label_name['ocnli']=entailment,neutral,contradiction
label_name['tnews']=100,101,102,103,104,106,107,108,109,110,112,113,114,115,116
label_name['cluewsc']=true,false

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ ! -f pai_chinese_bpe.model ]; then
  wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/rapidformer/zeroclue/pai_chinese_bpe.model
fi

VOCAB_FILE=pai_chinese_bpe.model
TOKENIZER=ChineseBPETokenizer

SEQ_LEN=2048
MAX_POSITION_EMBEDDINGS=2048
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
LOG_INTERVAL=1
EVAL_ITERS=10
EVAL_INTERVAL=1000000
SAVE_INTERVAL=10000000
TASK=csldcp
TRAIN_DATA=train_${TASK}.tsv
VALID_DATA=dev_${TASK}.tsv
NUM_EXPERTS="32 32 32 32 32 32 32 32 32 32 64 64"
PRETRAIN_CHECKPOINT_PATH=ckpts
SAVE_CHECKPOINT_PATH=/mnt/finetune_${TASK}
rapidformer_options="  \
        --pretrained-model-name-or-path ${PRETRAIN_CHECKPOINT_PATH} \
        --save ${SAVE_CHECKPOINT_PATH} \
        --tokenizer-type ${TOKENIZER} \
        --vocab-file ${VOCAB_FILE} \
        --tensor-model-parallel-size 1 \
        --train-data $TRAIN_DATA \
        --valid-data $VALID_DATA \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --pattern ${pattern[$TASK]} \
        --label_desc ${label_desc[$TASK]} \
        --label_enumerate_values ${label_name[$TASK]} \
        --input_schema ${input_schema[$TASK]} \
        --keep-last \
        --micro-batch-size 16 \
        --epochs 3 \
        --lr 1e-4 \
        --lr-decay-style cosine \
        --lr-warmup-fraction 0.1 \
        --min-lr 1e-6 \
        --num-workers 0 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.014 \
        --override-lr-scheduler \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --DDP-impl local \
        --mixed-precision \
        --no-contiguous-buffers-in-local-ddp \
        --hysteresis 2 \
        --eod-mask-loss \
        --checkpoint-activations \
        --disable-moe-token-dropping \
        --top-k-linear-strategy normal \
        --num-experts ${NUM_EXPERTS} \
        --mlp-type residual \
		    --zero-2-memory-optimization \
        --moe-loss-coeff 0.01 \
        "
run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS finetune_gpt_moe.py ${rapidformer_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x


