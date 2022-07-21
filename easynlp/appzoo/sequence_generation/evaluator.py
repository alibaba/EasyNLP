import numpy as np
from rouge import Rouge
import torch
from tqdm import tqdm
import torch.utils.data.dataloader as DataLoader

class SequenceGenerationEvaluator(object):
    def __init__(self, valid_dataset,**kwargs):
        self.valid_dataset = valid_dataset
        self.valid_loader=DataLoader.DataLoader(self.valid_dataset)
        self.best_valid_score=0

    def evaluate(self, model):
        y_preds = list()
        y_trues = list()
        for i, batch in enumerate(tqdm(self.valid_loader)):
            for key in batch:
                if key=='src_text' or key=='tgt_text':
                    continue
                batch[key]=torch.cat(batch[key]).cuda().unsqueeze(0)

            with torch.no_grad():
                gen = model.generate(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"],
                                     num_beams=1,
                                     min_length=8,
                                     max_length=40,
                                     early_stopping=True,
                                     no_repeat_ngram_size=2,
                                     num_return_sequences=1,
                                     decoder_start_token_id=model._tokenizer.cls_token_id,
                                     eos_token_id=model._tokenizer.sep_token_id)
            pred_tmp=[model._tokenizer.decode(t, skip_special_tokens=True) for t in gen]
            true_tmp=batch['tgt_text']
            if model._is_zh==True:
                y_preds.extend([" ".join(pred_tmp[0].replace(" ", ""))])
                y_trues.extend([" ".join(true_tmp[0].replace(" ", ""))])
            else:
                y_preds.extend(pred_tmp)
                y_trues.extend(true_tmp)
        y_preds = [t if t.strip() else "" for t in y_preds]
        rouge = Rouge()
        scores = rouge.get_scores(y_preds, y_trues, avg=True)
        print("Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f}".format(
            scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100,
            scores['rouge-l']['f'] * 100))
        rst = [('rouge-l', scores['rouge-l']['f'] * 100),
               ('rouge-1', scores['rouge-1']['f'] * 100),
               ('rouge-2', scores['rouge-2']['f'] * 100), ]
        return rst



