import json
import torch
from threading import Lock
from ..dataset import BaseDataset
from ...modelzoo.models.bert import BertTokenizerFast

class InformationExtractionDataset(BaseDataset):

    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 input_schema,
                 max_seq_length,
                 *args,
                 **kwargs):

        super(InformationExtractionDataset, self).__init__(data_file,
                                                           input_schema=input_schema,
                                                           output_format="dict",
                                                           *args,
                                                           **kwargs)

        self.max_seq_length = max_seq_length
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)

    def convert_single_row_to_example(self, row):

        id = row[self.column_names[0]]
        instruction = row[self.column_names[1]][2:-2]
        start = row[self.column_names[2]][1: -1]
        if start == "":
            start = []
        else:
            start = start.split(",")
            start = [int(i) for i in start]
        end = row[self.column_names[3]][1: -1]
        if end == "":
            end = []
        else:
            end = end.split(",")
            end = [int(i) for i in end]
        target = row[self.column_names[4]]

        example = self.tokenizer(
                instruction,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_offsets_mapping=True
            )
            
        example["id"] = id
        example["instruction"] = instruction
        example["start"] = start
        example["end"] = end
        example["target"] = target

        return example

    def batch_fn(self, features):

        batch = []
        for f in features:
            batch.append({'input_ids': f['input_ids'],
                          'token_type_ids': f['token_type_ids'],
                          'attention_mask': f['attention_mask']})
        
        batch = self.tokenizer.pad(
            batch,
            padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_seq_length,
            return_tensors="pt"
        )

        labels = torch.zeros(len(features), 1, self.max_seq_length, self.max_seq_length)  # 阅读理解任务entity种类为1 [bz, 1, max_len, max_len]
        for feature_id, feature in enumerate(features): # 遍历每个样本
            starts, ends = feature['start'], feature['end']
            offset = feature['offset_mapping'] # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
            position_map = {}
            for i, (m, n) in enumerate(offset):
                if i != 0 and m == 0 and n == 0:
                    continue
                for k in range(m, n + 1):
                    position_map[k] = i # 字符级别的第k个字符属于分词i
            for start, end in zip(starts, ends):
                end -= 1
                # MRC 没有答案时则把label指向CLS
                if start == 0:
                    assert end == -1
                    labels[feature_id, 0, 0, 0] = 1
                else:
                    if start in position_map and end in position_map:
                        # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                        labels[feature_id, 0, position_map[start], position_map[end]] = 1
            
        batch["label_ids"] = labels

        tempid = []
        tempinstruction = []
        tempoffset_mapping = []
        temptarget = []
        for i in range(len(features)):
            tempid.append(features[i]["id"])
            tempinstruction.append(features[i]["instruction"])
            tempoffset_mapping.append(features[i]["offset_mapping"])
            temptarget.append(features[i]["target"])
            
        batch["id"] = tempid
        batch["instruction"] = tempinstruction
        batch["offset_mapping"] = tempoffset_mapping
        batch["target"] = temptarget

        return batch