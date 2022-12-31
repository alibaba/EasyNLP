from time import time
from datetime import timedelta

from fastNLP.io.loader import JsonLoader
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.pipe.pipe import Pipe
from fastNLP.core.const import Const

class MatchSumLoader(JsonLoader):
    
    def __init__(self, candidate_num, encoder, max_len=511):
        fields = {'text_id': 'text_id',
             'candidate_id': 'candidate_id',
               'summary_id': 'summary_id'
             }
        super(MatchSumLoader, self).__init__(fields=fields)
        
        self.candidate_num = candidate_num
        self.max_len = max_len
        self.encoder = encoder

        if encoder == 'bert':
            self.sep_id = [102] # '[SEP]' (BERT)
        else:
            self.sep_id = [2] # '</s>' (RoBERTa)

    def _load(self, path):
        dataset = super(MatchSumLoader, self)._load(path)
        return dataset
    
    def load(self, paths):        
        
        def get_seq_len(instance):
            return len(instance['text_id'])
        
        def sample(instance, candidate_num):
            candidate_id = instance['candidate_id'][:candidate_num]
            return candidate_id
        
        def truncate_candidate_id(instance, max_len):
            candidate_id = []
            for i in range(len(instance['candidate_id'])):
                if len(instance['candidate_id'][i]) > max_len:
                    cur_id = instance['candidate_id'][i][:(max_len - 1)]
                    cur_id += self.sep_id
                else:
                    cur_id = instance['candidate_id'][i]
                candidate_id.append(cur_id)
            return candidate_id

        print('Start loading datasets !!!')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])
            
            if name == 'train':
                datasets[name].apply(lambda ins: truncate_candidate_id(ins, self.max_len), new_field_name='candidate_id')
            
            # set input and target
            datasets[name].set_input('text_id', 'candidate_id', 'summary_id')
            #datasets[name].set_input('text_id', 'candidate_id', 'summary_id', 'summary_sent_id','idxIn_ext_idx', 'ext_sents')

            # set padding value
            if self.encoder == 'bert':
                pad_id = 0
            else:
                pad_id = 1 # for RoBERTa
            datasets[name].set_pad_val('text_id', pad_id)
            datasets[name].set_pad_val('candidate_id', pad_id)
            datasets[name].set_pad_val('summary_id', pad_id)
            # datasets[name].set_pad_val('summary_sent_id', pad_id)
            # datasets[name].set_pad_val('candidate_sent_id', pad_id)

        print('Finished in {}'.format(timedelta(seconds=time()-start)))

        return DataBundle(datasets=datasets)

class MatchSumPipe(Pipe):

    def __init__(self, candidate_num, encoder):
        super(MatchSumPipe, self).__init__()
        self.candidate_num = candidate_num
        self.encoder = encoder

    def process(self, data_bundle):

        return data_bundle
        
    def process_from_file(self, paths):
        data_bundle = MatchSumLoader(self.candidate_num, self.encoder).load(paths)
        return self.process(data_bundle)

