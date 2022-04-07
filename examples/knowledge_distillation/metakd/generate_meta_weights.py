import sys
import numpy as np
import scipy.spatial.distance as distance
from tqdm import tqdm


def read_embeddings(input_str):
    outcome = np.zeros(shape=embedding_size)
    items = input_str.split(' ')
    for i in range(embedding_size):
        outcome[i] = float(items[i])
    return outcome


def compute_weight(domain, current_embedding, centroid_embeddings):
    current_centroid = centroid_embeddings[domain]
    other_centroids = list()
    for label in centroid_embeddings.keys():
        if label != domain:
            other_centroids.append(centroid_embeddings[label])
    other_centroids = np.array(other_centroids)
    other_centroid_mean = np.mean(other_centroids, axis=0)
    first_cos_sim = 1 - distance.cosine(current_embedding, current_centroid)
    second_cos_sim = 1 - distance.cosine(current_embedding, other_centroid_mean)
    return (first_cos_sim + second_cos_sim) / 2


# centroid extraction
dataset_file = sys.argv[1]
output_dataset_path = sys.argv[2]
domains = sys.argv[3].split(",")
# labels = ['books', 'dvd', 'electronics', 'kitchen']
embedding_size = 768

domain_embeddings = dict()
for domain in domains:
    domain_embeddings[domain] = list()

print("Computing centroid embedding...")
with open(dataset_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        if i == 0:
            continue
        line = line.strip()
        items = line.split('\t')
        domain = items[-2]
        embeddings = read_embeddings(items[-1])
        domain_embeddings[domain].append(embeddings)

centroid_embeddings = dict()
for label in domain_embeddings:
    data_embeddings = np.array(domain_embeddings[label])
    # print(data_embeddings.shape)
    centroid_embeddings[label] = np.mean(data_embeddings, axis=0)

# weight computation
print("Computing weights....")
output_data = list()
fout = open(output_dataset_path, "w")
fout.write("\t".join(["guid", "text_a", "text_b", "label", "domain", "weight"]) + "\n")
with open(dataset_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(tqdm(lines)):
        if i == 0:
            continue
        line = line.strip()
        items = line.split('\t')

        domain = items[-2]
        embeddings = read_embeddings(items[-1])
        weight = compute_weight(domain, embeddings, centroid_embeddings)
        fout.write("\t".join(items[:5] + [str(weight)]) + "\n")
fout.close()