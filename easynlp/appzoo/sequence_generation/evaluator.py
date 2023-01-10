import numpy as np
from rouge import Rouge
import torch
from tqdm import tqdm
import torch.utils.data.dataloader as DataLoader
import os
import json

class SequenceGenerationEvaluator(object):
    def __init__(self, valid_dataset, user_defined_parameters, **kwargs):
        self.valid_dataset = valid_dataset
        self.valid_loader=DataLoader.DataLoader(self.valid_dataset)
        self.best_valid_score=0

        pretrained_model_name_or_path = kwargs.get('pretrained_model_name_or_path')
        pretrained_model_name_or_path = pretrained_model_name_or_path if pretrained_model_name_or_path else kwargs.get('few_shot_anchor_args').pretrained_model_name_or_path
        # if 'pretrained_model_name_or_path' in kwargs:
        #     pretrained_model_name_or_path = kwargs['pretrained_model_name_or_path']
        # else:
        #     few_shot_anchor_args = kwargs['few_shot_anchor_args']
        #     pretrained_model_name_or_path = few_shot_anchor_args.pretrained_model_name_or_path
        
        if user_defined_parameters is not None:
            if type(user_defined_parameters)=='str':
                self.user_defined_parameters=json.loads(user_defined_parameters)
            else:
                self.user_defined_parameters=user_defined_parameters
        else:
            self.user_defined_parameters={}
        
        if os.path.exists(pretrained_model_name_or_path):
            local_path=pretrained_model_name_or_path
        else:
            raise FileNotFoundError('The provided model path %s does not exist, please check.' % pretrained_model_name_or_path)

        self.config_path=local_path+'/config.json'
        with open(self.config_path, 'r') as load_f:
            load_dict = json.load(load_f)
            self.decoder_only = 'gpt2' in pretrained_model_name_or_path or ("architectures" not in load_dict) or ("architectures" in load_dict and 'bloom' == load_dict.get('model_type', ''))
            self.is_randeng = 'randeng' in pretrained_model_name_or_path or 'randeng' == load_dict.get('model_type', '')

        self.max_encoder_length = kwargs.get('max_encoder_length', int(self.user_defined_parameters.get("max_encoder_length", 512)))
        self.min_decoder_length = int(self.user_defined_parameters.get("min_decoder_length", 8))
        self.max_decoder_length = int(self.user_defined_parameters.get("max_decoder_length", 128))
        self.no_repeat_ngram_size = int(self.user_defined_parameters.get("no_repeat_ngram_size", 2))
        self.num_beams = int(self.user_defined_parameters.get("num_beams", 5))
        self.num_return_sequences = int(self.user_defined_parameters.get("num_return_sequences", 5))

    def evaluate(self, model):
        model.eval()
        y_preds = list()
        y_trues = list()
        for i, batch in enumerate(tqdm(self.valid_loader)):
            for key in batch:
                if key=='src_text' or key=='tgt_text':
                    continue
                batch[key]=torch.cat(batch[key]).cuda().unsqueeze(0)

            if self.decoder_only:
                input_len = batch["attention_mask"][0].sum().item()
                max_decoder_length = self.max_decoder_length + input_len
            else:
                max_decoder_length = self.max_decoder_length
            if self.is_randeng:
                eos_token_id = model._tokenizer.eos_token_id
            else:
                eos_token_id = model._tokenizer.sep_token_id
            gen = model.generate(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    num_beams=1,
                                    min_length=self.min_decoder_length,
                                    max_length=max_decoder_length,
                                    early_stopping=True,
                                    no_repeat_ngram_size=self.no_repeat_ngram_size,
                                    num_return_sequences=1,
                                    decoder_start_token_id=model._tokenizer.cls_token_id,
                                    eos_token_id=eos_token_id)
            if self.decoder_only:
                pred_tmp=[model._tokenizer.decode(t[batch["attention_mask"][0].sum().item():], skip_special_tokens=True) for t in gen]
            else:
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
        # open('./result.txt','a+').write("Rouge 1/2/L: {:.2f}/{:.2f}/{:.2f} \n".format(
            # scores['rouge-1']['f'] * 100, scores['rouge-2']['f'] * 100,
            # scores['rouge-l']['f'] * 100))
        rst = [('rouge-l', scores['rouge-l']['f'] * 100),
               ('rouge-1', scores['rouge-1']['f'] * 100),
               ('rouge-2', scores['rouge-2']['f'] * 100), ]
        return rst



