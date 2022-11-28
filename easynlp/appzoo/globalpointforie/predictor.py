import json
import torch
import numpy as np
from .evaluator import get_predict_result
from ...modelzoo.models.bert import BertTokenizerFast

#假定目前的输入样式每行只有一个句子的一个任务（命名实体识别NER、关系抽取RE、事件抽取EE）
#对于命名实体识别，scheme = 时间;选手;……
#对于关系抽取、事件抽取，scheme = {'竞赛名称':'主办方;承办方',
#                       '比赛':'冠军;亚军']}

class GlobalPointForIEPredictor(object):

   def __init__(self, model_dir, model_cls, *args, **kwargs):
         super(GlobalPointForIEPredictor, self).__init__()

         self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
         self.model = model_cls.from_pretrained(model_dir)
         if torch.cuda.is_available():
            self.model = self.model.cuda()
         self.model.eval()
         self.input_file = kwargs.pop("input_file")
         self.output_file = kwargs.pop("output_file")
         self.max_seq_length = kwargs.pop("max_seq_length")

         assert self.input_file is not None
   
   #将一句话的所有真正的命名实体识别任务转换为batch，方便进行模型进行训练
   def get_ner_batch(self, id, content, entity_types):

      examples = []

      temp_id = []
      temp_entity_types = []
      temp_instruction = []
      temp_offset_mapping = []

      for i in range(len(entity_types)):
         instruction = "找到文章中所有【{}】类型的实体？文章：【{}】".format(entity_types[i], content)
         example = self.tokenizer(
                     instruction,
                     truncation=True,
                     max_length=self.max_seq_length,
                     padding="max_length",
                     return_offsets_mapping=True)

         temp_id.append(id)
         temp_entity_types.append(entity_types[i])
         temp_instruction.append(instruction)
         temp_offset_mapping.append(example["offset_mapping"])

         examples.append(example)

      batch = []
      for f in examples:
         batch.append({'input_ids': f['input_ids'],
                       'token_type_ids': f['token_type_ids'],
                       'attention_mask': f['attention_mask']})
                          
      batch = self.tokenizer.pad(
                  batch,
                  padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
                  max_length=self.max_seq_length,
                  return_tensors="pt")
         
      batch["id"] = temp_id
      batch["entity_type"] = temp_entity_types
      batch["instruction"] = temp_instruction
      batch["offset_mapping"] = temp_offset_mapping

      return batch
   
   def get_predict_result(self, batchs, probs, indices, max_seq_length):

      probs = probs.squeeze(1)  # topk结果的概率
      indices = indices.squeeze(1)  # topk结果的索引
      answers = []

      for id, instruction, offset_mapping, prob, index in zip(batchs["id"], batchs["instruction"], batchs["offset_mapping"], probs, indices):

         answer = []
         index_ids = torch.Tensor([i for i in range(len(index))]).long()
         entity_index = index[prob > 0.6]
         index_ids = index_ids[prob > 0.6]
         for ei, entity in enumerate(entity_index):

            entity = entity.cpu().numpy()
            start_end = np.unravel_index(entity, (max_seq_length, max_seq_length))

            s = offset_mapping[start_end[0]][0]
            e = offset_mapping[start_end[1]][1]
            ans = instruction[s: e]

            answer.append({'ans':ans, 'prob': float(prob[index_ids[ei]]), 'pos': [s, e]})
         
         answers.append(answer)
      
      return answers
   
   def run_model(self, batch):

      try:
         batch = {
            key: val.cuda() if isinstance(val, torch.Tensor) else val
            for key, val in batch.items()}
      except RuntimeError:
         batch = {key: val for key, val in batch.items()}

      with torch.no_grad():
         outputs = self.model(batch)
      topk_probs = outputs["topk_probs"]
      topk_indices = outputs["topk_indices"]

      return topk_probs, topk_indices
   
   def print_(self, answers):
      with open(self.output_file, 'a+', encoding='utf-8') as fw:
         json.dump(answers, fw, ensure_ascii=False, indent=2)
   
   def _hand_answers(self, batch, answers):

      all_hand_answers = list()
      entity_instance = {}
      for i in range(len(answers)):
         if batch["entity_type"][i] not in entity_instance:
            entity_instance[batch["entity_type"][i]] = []
         for j in range(len(answers[i])):
            temp = {}
            temp["text"] = answers[i][j]["ans"]
            temp["probability"] = answers[i][j]["prob"]
            temp["start"] = answers[i][j]["pos"][0]
            temp["end"] = answers[i][j]["pos"][1]
            entity_instance[batch["entity_type"][i]].append(temp)
      all_hand_answers.append(entity_instance)

      return all_hand_answers


   def run(self):

      lines = json.load(open(self.input_file, encoding='utf8'))

      for line in lines:
         
         id = line["id"]
         task = line["task"]
         scheme = line["scheme"]
         content = line["content"]

         if task == "NER":

            entity_types = scheme.split(";")
            batch = self.get_ner_batch(id, content, entity_types)
            probs, indices = self.run_model(batch)
            answers = self.get_predict_result(batch, probs, indices, self.max_seq_length)
            answers = self._hand_answers(batch, answers)
            self.print_(answers)
            
         else:

            entity_types = [i for i in scheme.keys()]
            batch = self.get_ner_batch(id, content, entity_types)
            probs, indices = self.run_model(batch)
            answers = self.get_predict_result(batch, probs, indices, self.max_seq_length)

            batch = self.get_ner_batch_2(id, content, answers, scheme)
            probs, indices = self.run_model(batch)
            answers_2 = self.get_predict_result(batch, probs, indices, self.max_seq_length)
            answers = self._hand_answers_2(batch, answers, answers_2)
            self.print_(answers)

   def _hand_answers_2(self, batch, answers, answers_2):

      all_hand_answers = list()
      instance = {}
      num = 0
      for i in range(len(answers)):
         if batch["entity_type"][i]["type"] not in instance:
            instance[batch["entity_type"][i]["type"]] = []
         for j in range(len(answers[i])):
            temp = {}
            temp["text"] = answers[i][j]["ans"]
            temp["probability"] = answers[i][j]["prob"]
            temp["start"] = answers[i][j]["pos"][0]
            temp["end"] = answers[i][j]["pos"][1]
            temp["relations"] = {}

      #开始看answers_2的答案
            
            for k in range(len(batch["entity_type"][i]["role"])):
               temp["relations"][batch["entity_type"][i]["role"][k]] = []

               if len(answers_2[num]) == 0:
                  num += 1
                  continue
               else:
                  for l in range(len(answers_2[num])):
                     temp_2 = {}
                     temp_2["text"] = answers_2[num][l]["ans"]
                     temp_2["probability"] = answers_2[num][l]["prob"]
                     temp_2["start"] = answers_2[num][l]["pos"][0]
                     temp_2["end"] = answers_2[num][l]["pos"][1]

                     temp["relations"][batch["entity_type"][i]["role"][k]].append(temp_2)
                     num += 1
               
            instance[batch["entity_type"][i]["type"]].append(temp)
      all_hand_answers.append(instance)

      return all_hand_answers
   
   def get_ner_batch_2(self, id, content, answers, scheme):
      
      examples = []

      temp_id = []
      temp_entity_types = {}
      temp_instruction = []
      temp_offset_mapping = []

      entity_types = [i for i in scheme.keys()]

      for i in range(len(answers)):
         temp_entity_types[i] = {}
         temp_entity_types[i]["type"] = entity_types[i]
         temp_entity_types[i]["role"] = []
         for j in range(len(answers[i])):
            identified_entity = answers[i][j]["ans"]

            relation_types = scheme[entity_types[i]].split(";")

            for k in range(len(relation_types)):
               
               instruction = "找到文章中【{}】的【{}】？文章：【{}】".format(identified_entity, relation_types[k], content)
               example = self.tokenizer(
                     instruction,
                     truncation=True,
                     max_length=self.max_seq_length,
                     padding="max_length",
                     return_offsets_mapping=True)

               temp_id.append(id)
               temp_entity_types[i]["role"].append(relation_types[k])
               temp_instruction.append(instruction)
               temp_offset_mapping.append(example["offset_mapping"])

               examples.append(example)

      batch = []
      for f in examples:
         batch.append({'input_ids': f['input_ids'],
                       'token_type_ids': f['token_type_ids'],
                       'attention_mask': f['attention_mask']})
                          
      batch = self.tokenizer.pad(
                  batch,
                  padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
                  max_length=self.max_seq_length,
                  return_tensors="pt")
               
      batch["id"] = temp_id
      batch["entity_type"] = temp_entity_types
      batch["instruction"] = temp_instruction
      batch["offset_mapping"] = temp_offset_mapping

      return batch