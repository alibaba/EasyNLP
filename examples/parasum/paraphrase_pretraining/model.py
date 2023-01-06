import torch
from torch import nn
from torch.nn import init
from collections import Counter
from transformers import BertModel, RobertaModel
from transformers import AutoModel, AutoTokenizer

class MatchSum(nn.Module):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        
        if encoder == 'bert':
            # self.encoder = BertModel.from_pretrained('D:\data\\bert_eng')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.encoder = RobertaModel.from_pretrained('roberta-base')
        # for n, p in self.encoder.named_parameters():
        #     p.require_grad=False
        in_features = 768
        self._dropout = torch.nn.Dropout(p=0.2)
        # self._classification_layer = torch.nn.Linear(in_features, 1)
        self._classification_layer = torch.nn.Linear(in_features, 1)
        self._sigmoid = nn.Sigmoid()
        torch.nn.init.xavier_uniform_(self._classification_layer.weight)
        torch.nn.init.constant_(self._classification_layer.bias, 0)

    def forward(self, candidate_id, label):
        
        batch_size = candidate_id.size(0)
        
        pad_id = 0     # for BERT
        # if text_id[0][0] == 0:
        #     pad_id = 1 # for RoBERTa

        # get document embedding
        #input_mask = ~(text_id == pad_id)
        #out = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        #doc_emb = out[:, 0, :]
        #assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        # input_mask = ~(summary_id == pad_id)
        # out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
        # summary_emb = out[:, 0, :]
        # assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        # summary_score = self._sigmoid(self._classification_layer(self._dropout(summary_emb)))

        # get candidate embedding
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)
        
        # get candidate score
        score = self._sigmoid(self._classification_layer(self._dropout(candidate_emb)))
        return {'score': score.squeeze(1), 'label': label}

