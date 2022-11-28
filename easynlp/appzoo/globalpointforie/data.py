import json
import torch
from ..dataset import BaseDataset
from ...modelzoo.models.bert import BertTokenizerFast

class GlobalPointForIEDataset(BaseDataset):

    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 *args,
                 **kwargs):

        super().__init__(data_file,
                         *args,
                         **kwargs)

        assert ".json" in data_file
        assert self.data_source == "local"

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.data_file = data_file
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
        self.data_rows = self._readlines_from_file()
    
    def __len__(self):
        return len(self.data_rows)
    
    def __del__(self):
        pass
    
    def _readlines_from_file(self):

        data_rows = []
        lines = json.load(open(self.data_file, encoding='utf8'))

        for line in lines:

            id = line['ID']
            instruction = line['instruction']
            target = line['target']
            start = line['start']
            
            new_start, new_end = [], []
            for t, entity_starts in zip(target, start):
                for s in entity_starts:
                    new_start.append(s)
                    new_end.append(s + len(t))
            start, end = new_start, new_end
            target = '|'.join(target)

            data_rows.append({'id': id,
                             'instruction': instruction,
                             'start': start,
                             'end': end,
                             'target': target})
        
        return data_rows

    def __getitem__(self, item):
        row = self.data_rows[item]
        return self.convert_single_row_to_example(row)

    def convert_single_row_to_example(self, row):

        example = self.tokenizer(
                row['instruction'],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_offsets_mapping=True
            )
        
        for key, value in row.items():
            example[key] = value

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