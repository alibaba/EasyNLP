import torch
import numpy as np
import pandas as pd

# for financial
checkpoint_path = "financial_concept.pth.best"
entity_file = "../../../tmp/finData/finEntity.csv"
saving_file = "../../../tmp/finData/finConceptEmbedding.npy"


def output_embedding(name):
    chkpnt = torch.load(checkpoint_path)
    embeddings = chkpnt['embeddings']
    objects = np.array(chkpnt['objects'])
    idx = np.where(objects == name)[0]
    v = embeddings[idx][0].tolist()
    print(name)
    print(v)
    return v


concept_to_embedding = {}
for name in ['产业链-产品', '行业-企业', '人', '位置']:
    concept_to_embedding[name] = output_embedding(name)

entity_df = pd.read_csv(entity_file)
entity_name_to_concept = {}
entity_id_to_concept = {}
entity_id_to_embedding = {}
for i in range(len(entity_df)):
    name_list = entity_df.iloc[i]['name_list'].split('|')
    label = entity_df.iloc[i]['label']
    index = entity_df.iloc[i]['index']
    if label in ['产品', '产业链']:
        concept = '产业链-产品'
    elif label in ['上市公司', '行业']:
        concept = '行业-企业'
    else:
        concept = label
    for name in name_list:
        entity_name_to_concept[name] = concept

    entity_id_to_concept[index] = concept
    entity_id_to_embedding[index] = concept_to_embedding[concept]

np.save(saving_file, entity_id_to_embedding)
print("successful save ", saving_file)





