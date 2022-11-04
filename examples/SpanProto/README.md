## SpanProto：Few-shot NER
This project is implemented for the EMNLP2022 (main conference) paper: "[SpanProto: A Two-stage Span-based Prototypical Network for Few-shot Named Entity Recognition](https://arxiv.org/pdf/2210.09049.pdf)". Our code is based on pytorch and huggingface transformers.

In this paper, we present a novel two-stage span-based prototypical network (SpanProto) for few-shot named entity recognition (NER). 

The main motivations are:
- Traditional few-shot NER methods ignore of the mentions boundary information, which is more crucial to NER;
- In few-shot NER, there are only k-shot entities of each class, but has many non-entity tokens (i.e., tagging "O") which may disturb the performance.

The main methods:
- We decompose the few-shot NER into two stages, including span extractor and mention classifier. 
- In the span extractor, we convert the original sequentail tagging into a matrix, where each element stands for one span (start and end position) and 1 means this span is an entity and 0 means not. We can train the span extractor via cross-entropy on support set, and recall all candidate spans on query set by model inference.
- In the mention classifier, we capture each span representations and train the model with protopical learning on support set. When inference on query set, we split out false negative spans via margin-based loss, and predict others class by calculating and sorting the distance between each span embeddings and prototype vectors.

---

## Data and Models

Please download Few-NERD through this link: [https://ningding97.github.io/fewnerd/](https://ningding97.github.io/fewnerd/), and move it under the dir: ```dataset/```. (You can directly download from [https://cloud.tsinghua.edu.cn/f/56fb277d3fd2437a8ee3/?dl=1](https://cloud.tsinghua.edu.cn/f/56fb277d3fd2437a8ee3/?dl=1), and unzip it by ```unzip episode-data.zip```.)


We recommend you directly downloading the backbone (e.g., [bert-base-uncased](https://huggingface.co/bert-base-uncased)) through huggingface, and move it to a new dir (e.g., ```pre-trained-lm/bert-base-uncased```).

## Code Runing
You can run this code by:
```bash
sh scripts/fewner/run_fewnerd.sh
```
In this script, you can define the following values:
- mode: the mode of Few-NERD. ("inter" or "intra")
- N: the number of entity classes (w/o. "O" type). (5 or 10)
- Q: the number of entities of each class in one episode support set. (1 or 5)
- K: the number of entities of each class in one episode query set. (1 or 5)

---

## Acknowledgement
This project is support by Alibaba Platform of AI (PAI).

```
@article{DBLP:journals/corr/abs-2210-09049,
  author    = {Jianing Wang and
               Chengyu Wang and
               Chuanqi Tan and
               Minghui Qiu and
               Songfang Huang and
               Jun Huang and
               Ming Gao},
  title     = {SpanProto: {A} Two-stage Span-based Prototypical Network for Few-shot
               Named Entity Recognition},
  journal   = {CoRR},
  volume    = {abs/2210.09049},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2210.09049},
  doi       = {10.48550/arXiv.2210.09049},
  eprinttype = {arXiv},
  eprint    = {2210.09049},
  timestamp = {Wed, 19 Oct 2022 12:47:31 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2210-09049.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

