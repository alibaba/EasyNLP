import torch
from ..dataset import BaseDataset
from ...modelzoo import AutoTokenizer

class OpenDomainDialogueDataset(BaseDataset):
    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_text_length,
                 max_label_length,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         *args,
                         **kwargs)
        
        self.max_text_length = max_text_length
        self.max_label_length = max_label_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    
    def convert_single_row_to_example(self, row):
        sentences = row.split('\t')
        episodes = []
        history = ''
        for turn in range(len(sentences) // 2):
            text = sentences[turn * 2].replace('\\n', '\n')
            text = history + text
            label = sentences[(turn * 2) + 1].replace('\\n', '\n')
            history = text + label
            if text == '__null__' or label == '__null__':
                break
            encoding = self.tokenizer(text,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_text_length)
            label_ids = self.tokenizer(label,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_label_length)['input_ids']
            encoding.update({'label_ids': label_ids})
            encoding.update({'label': label_ids})
            episodes.append(encoding)
        return episodes

    def batch_fn(self, features):
        refactor_feat = [ep for session in features for ep in session]
        return {k: torch.tensor([dic[k] for dic in refactor_feat]) for k in refactor_feat[0]}
