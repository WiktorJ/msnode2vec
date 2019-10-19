import click
import numpy as np
from distance import principal_angle_distance, cos_distance
from utils import map_embeddings_to_consecutive
from sklearn.metrics.pairwise import cosine_similarity


# @click.command()
# @click.option("-e1", "--embeddings_1", type=str, required=True)
# @click.option("-e2", "--embeddings_2", type=str, required=True)
def pa(embeddings_1, embeddings_2):
    embs = map_embeddings_to_consecutive([embeddings_1, embeddings_2])
    # principal_angle_distance(embs)
    cos_sim = []
    for i in range(len(embs[0])):
        cos_sim.append(cos_distance(embs[0][i], embs[1][i]))
    # print(sorted(cos_sim))
    e_max = embs[0].max()
    e_min = embs[0].min()
    control = (e_max - e_min) * np.random.random(embs[0].shape) + e_min
    c = []
    for i in range(len(embs[0])):
        c.append(cos_distance(embs[1][i], control[i]))
    # print(sorted(c))
    print(np.std(cos_sim))
    print(np.var(cos_sim))
    print(min(cos_sim))
    print(max(cos_sim))
    print("-------")
    print(np.std(c))
    print(np.var(c))
    print(min(c))
    print(max(c))

# (b - a) * random_sample() + a

if __name__ == '__main__':
    pa('/Users/w.jurasz/Studies/GR/node2vec/emb/lesmis_base2.emb',
       '/Users/w.jurasz/Studies/GR/node2vec/emb/lesmis_base.emb')
    # pa()
