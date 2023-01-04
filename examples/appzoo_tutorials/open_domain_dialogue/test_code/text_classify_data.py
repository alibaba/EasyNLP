import random
from torch.utils.data import DataLoader, RandomSampler
from easynlp.appzoo import ClassificationDataset

pretrained_model_name_or_path = '/apsarapangu/disk3/xianyu.lzy/.easynlp/modelzoo/public/bert-base-uncased'
tables = 'train.tsv'
max_seq_len = 128
input_schema = 'label:str:1,sid1:str:1,sid2:str:1,sent1:str:1,sent2:str:1'
first_sequence = 'sent1'
second_sequence = 'sent2'
label_name = 'label'
label_enumerate_values = '0,1'
user_defined_parameters = {'pretrain_model_name_or_path': 'bert-base-uncased', 'app_parameters': {}}

dataset = ClassificationDataset(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    data_file=tables,
    max_seq_length=max_seq_len,
    input_schema=input_schema,
    first_sequence=first_sequence,
    second_sequence=second_sequence,
    label_name=label_name,
    label_enumerate_values=label_enumerate_values,
    user_defined_parameters=user_defined_parameters,
    is_training=True
)

d_len = len(dataset)
print(d_len)
idx = random.randint(0, d_len)
item = dataset[idx]
print(idx)
print(item)

sampler = RandomSampler(dataset)
loader = DataLoader(dataset,
                    sampler=sampler,
                    batch_size=32,
                    collate_fn=dataset.batch_fn,
                    num_workers=10)

for _step, batch in enumerate(loader):
    print(_step)
    print(batch)
    if _step >= 3:
        break