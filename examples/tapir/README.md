# TAPIR: Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning

![image](https://github.com/user-attachments/assets/1c48a0ce-bc47-468a-9762-c952b1494d0e)


## 📖Introduction

Our paper "Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning" introduces a framework called Task-Aware Curriculum Planning for Instruction Refinement (TAPIR). TAPIR is designed to improve the instruction-following capabilities of large language models (LLMs) by addressing the challenges of task distribution and instruction difficulty during training. The framework uses an oracle LLM to select difficult instructions for a student LLM and adjusts task distributions to balance the student's capabilities. TAPIR also incorporates curriculum planning to escalate task difficulty levels progressively.

## 🧠Models

Download Tapir 7B:

```
bash dl_tapir_7B.sh
```

**Please use official Llama2 template:**

>[INST] \<\<SYS>> {{ .System }} \<\</SYS>>
>
>{{ .Prompt }}
>
>[/INST]

## 🗃️Data

Download Tapir_Instruct_70k Dataset:

https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/tutorials/TAPIR-Distillation/Tapir_Instruct.json

## 📜 Citation

If you find our work helpful, please cite it!

```
@misc{TAPIR,
      title={Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning}, 
      author={Yuanhao Yue and Chengyu Wang and Jun Huang and Peng Wang},
      year={2024},
      eprint={2405.13448},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.13448}, 
}
```
