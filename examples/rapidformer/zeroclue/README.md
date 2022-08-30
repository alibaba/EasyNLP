# ZeroClue with Alibaba PAI's GPT-MoE

## Download Docker Image

```[bash]
docker pull pai-image-manage-registry.cn-shanghai.cr.aliyuncs.com/pai/easy_nlp:0.0.7
```

## Download Chinese GPT MoE Pretrained Model

```[bash]
./ossutil64 cp -r oss://atp-modelzoo-sh/tutorial/rapidformer/zeroclue/ckpts ckpts
```

## Domain Adaption

```[bash]
sh run_finetune_gpt_moe.sh
```

## Zero Shot Prediction

```[bash]
sh run_predict_gpt_moe.sh
```