# ARoBERT

We have release the code for our TASLP paper [ARoBERT: An ASR Robust Pre-trained Language Model for Spoken Language Understanding](https://ieeexplore.ieee.org/document/9721159/). Refer to [Here](https://github.com/alibaba/EasyTransfer/tree/master/scripts/pretraining_asr_robust_bert).

Spoken Language Understanding (SLU) aims to interpret the meanings of human speeches in order to support various human-machine interaction systems. A key technique for SLU is Automatic Speech Recognition (ASR), which transcribes speech signals into text contents. As the output texts of modern ASR systems unavoidably contain errors, mainstream SLU models either trained or tested on texts transcribed by ASR systems would not be sufficiently error robust. We present ARoBERT, an ASR Robust BERT model, which can be fine-tuned to solve a variety of SLU tasks with noisy inputs. To guarantee the robustness of ARoBERT, during pretraining, we decrease the fluctuations of language representations when some parts of the input texts are replaced by homophones or synophones. Specifically, we propose two novel self-supervised pre-training tasks for ARoBERT, namely Phonetically-aware Masked Language Modeling (PMLM) and ASRModel-adaptiveMasked LanguageModeling (AMMLM). The PMLM task explicitly fuses the knowledge of word phonetic similarities into the pre-training process, which forces homophones and synophones to share similar representations. In AMMLM, a data-driven algorithm is further introduced to mine typical ASR errors such that ARoBERT can tolerate ASR model errors. In the experiments, we evaluate ARoBERT over multiple datasets. The results show the superiority of ARoBERT, which consistently outperforms strong baselines. We have also shown that ARoBERT outperforms state-of-the-arts on a public benchmark. Currently, ARoBERT has been deployed in an online production system with significant improvements.


## Citation
Our paper citation is:

```
@article{taslp2022,
  author    = {Chengyu Wang and
               Suyang Dai and
               Yipeng Wang and
               Fei Yang and
               Minghui Qiu and
               Kehan Chen and
               Wei Zhou and
               Jun Huang},
  title     = {ARoBERT: An ASR Robust Pre-trained Language Model for Spoken Language Understanding},
  journal   = {{IEEE} {ACM} Trans. Audio Speech Lang. Process.},
  volume    = {30},
  pages     = {1207--1218},
  year      = {2022}
}
```
