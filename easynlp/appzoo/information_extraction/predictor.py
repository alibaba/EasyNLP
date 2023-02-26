import json
import torch
import numpy as np
from threading import Lock
from ...core.predictor import Predictor
from .evaluator import get_predict_result
from ...modelzoo.models.bert import BertTokenizerFast
from ...core.predictor import Predictor, get_model_predictor

#假定一个文件中只有一种任务，NER或RE或EE
#对于NER任务，scheme为xx；xx；xx，
#输出形式为id content q_and_a 注意，q_and_a的形式为[[实体类型 答案 答案开始位置 答案结束位置 答案的可能性大小]]
#对于非NER任务，scheme为竞赛名称：主办方，承办方；比赛：冠军，亚军
#输出形式为id content 竞赛名称 答案 答案开始位置 答案结束位置 答案的可能性大小
#以及id content  q_and_a 注意，q_and_a的形式为[[竞赛名称（xx）-主办方 答案 答案开始位置 答案结束位置 答案的可能性大小]]

class InformationExtractionPredictor(Predictor):

   def __init__(self, model_dir, model_cls, *args, **kwargs):
      super(InformationExtractionPredictor, self).__init__()

      self.MUTEX = Lock()

      self.task = kwargs["user_defined_parameters"]["task"]
      self.max_seq_length = kwargs.get("sequence_length")
      self.input_schema = kwargs.get("input_schema")
      self.column_names = [t.split(":")[0] for t in self.input_schema.split(",")]

      self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
      self.model_predictor = get_model_predictor(
         model_dir=model_dir,
         model_cls=model_cls,
         input_keys=[("input_ids", torch.LongTensor),
                     ("attention_mask", torch.LongTensor),
                     ("token_type_ids", torch.LongTensor)
                     ],
         output_keys=["topk_probs", "topk_indices"]
      )

   def preprocess(self, in_data):
      if not in_data:
         raise RuntimeError("Input data should not be None.")

      if not isinstance(in_data, list):
            in_data = [in_data]
      
      rst = {
         "id": [],
         "scheme": [],
         "content": [],
         "entity_type": [],
         "instruction": [],
         "offset_mapping": [],
         "input_ids": [],
         "token_type_ids": [],
         "attention_mask": []
      }

      for record in in_data:

         id = record[self.column_names[0]]
         scheme = record[self.column_names[1]]
         content = record[self.column_names[2]]

         if self.task == "NER":
            entity_types = scheme.split(";")
         else:
            entity_types =  [t.split(":")[0] for t in scheme.split(";")]
         
         try:
            self.MUTEX.acquire()
            examples = []
            for i in range(len(entity_types)):
               instruction = "找到文章中所有【{}】类型的实体？文章：【{}】".format(entity_types[i], content)
               example = self.tokenizer(
                     instruction,
                     truncation=True,
                     max_length=self.max_seq_length,
                     padding="max_length",
                     return_offsets_mapping=True)
               example["id"] = id
               example["scheme"] = scheme
               example["content"] = content
               example["entity_type"] = entity_types[i]
               example["instruction"] = instruction
               examples.append(example)
         finally:
            self.MUTEX.release()

         for e_index, example in enumerate(examples):
            rst["id"].append(example["id"])
            rst["scheme"].append(example["scheme"])
            rst["content"].append(example["content"])
            rst["entity_type"].append(example["entity_type"])
            rst["instruction"].append(example["instruction"])
            rst["offset_mapping"].append(example["offset_mapping"])
            rst["input_ids"].append(example["input_ids"])
            rst["token_type_ids"].append(example["token_type_ids"])
            rst["attention_mask"].append(example["attention_mask"])
      
      return rst
            
   def predict(self, in_data):

      answers = self.model_predictor.predict(in_data)
      answers = self.get_predict_result(answers, self.max_seq_length)

      if self.task == "NER":
         return [in_data, answers]
      else:

         rst = {
            "id": [],
            "content": [],
            "instruction": [],
            "question": [],
            "offset_mapping": [],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
         }

         try:
            self.MUTEX.acquire()
            examples = []
            for i in range(len(answers)):
               
               id = in_data["id"][i]
               scheme = in_data["scheme"][i]
               types = {t.split(":")[0]:t.split(":")[1].split(",") for t in scheme.split(";")}
               content = in_data["content"][i]
               entity_type = in_data["entity_type"][i]
               for j in range(len(answers[i])):
                  identified_entity = answers[i][j]["ans"]
                  
                  for k in range(len(types[entity_type])):
                     instruction = "找到文章中【{}】的【{}】？文章：【{}】".format(identified_entity, types[entity_type][k], content)
                     example = self.tokenizer(
                        instruction,
                        truncation=True,
                        max_length=self.max_seq_length,
                        padding="max_length",
                        return_offsets_mapping=True)
                     example["id"] = id
                     example["content"] = content
                     example["instruction"] = instruction
                     example["question"] = "{}({})-{}".format(entity_type, identified_entity, types[entity_type][k])
                     examples.append(example)
         finally:
            self.MUTEX.release()
         
         for e_index, example in enumerate(examples):
            rst["id"].append(example["id"])
            rst["content"].append(example["content"])
            rst["instruction"].append(example["instruction"])
            rst["question"].append(example["question"])
            rst["offset_mapping"].append(example["offset_mapping"])
            rst["input_ids"].append(example["input_ids"])
            rst["token_type_ids"].append(example["token_type_ids"])
            rst["attention_mask"].append(example["attention_mask"])
      
         answers = self.model_predictor.predict(rst)
         answers = self.get_predict_result(answers, self.max_seq_length)

         return [rst, answers]

   def get_predict_result(self, lists, max_seq_length):

      probs = lists["topk_probs"].squeeze(1)  # topk结果的概率
      indices = lists["topk_indices"].squeeze(1)  # topk结果的索引
      answers = []

      for id, instruction, offset_mapping, prob, index in zip(lists["id"], lists["instruction"], lists["offset_mapping"], probs, indices):

         answer = []
         index_ids = torch.Tensor([i for i in range(len(index))]).long()
         entity_index = index[prob > 0.6]
         index_ids = index_ids[prob > 0.6]
         for ei, entity in enumerate(entity_index):

            start_end = np.unravel_index(entity, (max_seq_length, max_seq_length))

            s = offset_mapping[start_end[0]][0]
            e = offset_mapping[start_end[1]][1]
            ans = instruction[s: e]

            answer.append({'ans':ans, 'prob': float(prob[index_ids[ei]]), 'pos': [s, e]})
         
         answers.append(answer)
      
      return answers
   
   def postprocess(self, data):

      in_data, answers = data[0], data[1]

      output_dict_list = []
      
      temp_id = ""
      for i in range(len(answers)):

         if in_data["id"][i] != temp_id:
            output_dict = {}
            output_dict["id"] = in_data["id"][i]
            output_dict["content"] = in_data["content"][i]
            output_dict["q_and_a"] = []
            temp_id = in_data["id"][i]
         for j in range(len(answers[i])):
            temp = []
            if self.task == "NER":
               temp.append(in_data["entity_type"][i]) #question
            else:
               temp.append(in_data["question"][i]) #question
            temp.append(answers[i][j]["ans"]) #answer
            temp.append(answers[i][j]["prob"]) #answer_prob
            temp.append(answers[i][j]["pos"][0]) #answer_start
            temp.append(answers[i][j]["pos"][1]) #answer_end
            output_dict["q_and_a"].append(temp)
         if in_data["id"][i] != temp_id or i == len(answers)-1:
            output_dict_list.append(output_dict)

      return output_dict_list
