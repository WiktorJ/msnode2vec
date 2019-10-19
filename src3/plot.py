import itertools

import numpy as np
from scipy import linalg
import matplotlib as mpl
import csv
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import cluster

from distance import map_clusters, get_gmm_clusters, calc_cluster_distance
from utils import get_as_numpy_array

G = nx.Graph()

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'black'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    # plt.xlim(-3., 2.)
    # plt.ylim(-2., 3.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


def get_different_assignments(labels1, labels2):
    s = []
    for i, tup in enumerate(zip(labels1, labels2)):
        if tup[0] != tup[1]:
            s.append(i)
    return s



# with open('../graph/lesmis.edgelist') as graph_file:
# with open('../graph/email-Eu-core-small-denominated.edgelist') as graph_file
with open('../graph/karate.edgelist') as graph_file:
# with open('../graph/email-Eu-core.txt') as graph_file:
    graph_csv = csv.reader(graph_file, delimiter=' ')
    for row in graph_csv:
        G.add_edge(int(row[0]), int(row[1]))

d = dict(nx.degree(G))

# embeddings = ["lesmis_base", "lesmis_biased"]
# titles = ["Lesmis graph", "lesmis_biased"]
embeddings = ["karate_base", "karate_biased"]
# titles = ["Lesmis graph", "lesmis_biased"]
# embeddings = ["email_base_loops", "email_biased_loops"]
clusters = 2

Xs = [get_as_numpy_array(f'../emb/{embedding}.emb') for embedding in embeddings]
predictions = [cluster.KMeans(n_clusters=clusters, random_state=0).fit(X).labels_ for X in Xs]
# predictions = [get_gmm_clusters(X, 6) for X in Xs]
# with open("../labels/email-Eu-core-department-labels-denominated.txt") as file:
#     labels = list(csv.reader(file, delimiter=' '))
# labels = [int(el[1]) for el in labels]
mapped_predictions = []
for pred in predictions[1:]:
    m = map_clusters(pred, predictions[0], clusters)
    mapped_predictions.append([m[el] for el in pred])
mapped_predictions = [predictions[0]] + mapped_predictions
# mapping1 = map_clusters(predictions[1], labels, 42)
# calc_cluster_distance(predictions[0], labels, 42, "base")
# calc_cluster_distance(predictions[1], labels, 42, "biased")

# predictions[0] = [mapping0[el] for el in predictions[0]]
# predictions[1] = [mapping1[el] for el in predictions[1]]


# print(kmeans.labels_)
# plot_results(X, prediction, gmm.means_, gmm.covariances_, 1,
#              'Bayesian Gaussian Mixture with a Dirichlet process prior')

diff = get_different_assignments(predictions[0], predictions[1])

# labels = {i: 'X' if i in diff else '' for i in range(len(G.nodes))}

pos = nx.spring_layout(G)
print(pos)
# plt.title(embeddings[0])
# nx.draw(G, pos=pos, node_list=d.keys(), node_size=[n * 2 for n in d.values()],
#         node_color=predictions[0], width=0.05, with_labels=True, font_size=8)
# plt.show(dpi=1500)
#
# plt.title(embeddings[0])
# nx.draw(G, pos=pos, node_list=d.keys(), node_size=[n * 2 for n in d.values()],
#         node_color=predictions[0], width=0.05)
# nx.draw_networkx_labels(G, pos, labels=labels)
# plt.show(dpi=1500)
print(f"estiated diff assigments: {len(diff) / len(G.nodes())}")
# plt.title("labels")
# nx.draw(G, pos=pos, node_list=d.keys(), node_size=0.2,
#         node_color=labels, width=0.0001)
# plt.show(dpi=1500)
# plt.savefig("labels.pdf")
for i in range(len(embeddings)):
    # plt.title(embeddings[i])
    nx.draw(G, pos=pos, node_list=d.keys(), node_size=[(n * 5) + 20 for n in d.values()],
            node_color=mapped_predictions[i], width=0.001)
    # plt.show(dpi=1500)
    plt.savefig(f"{embeddings[i]}.pdf")

# plt.title(embeddings[1])
# nx.draw(G, pos=pos, node_list=d.keys(), node_size=0.2,
#         node_color=predictions[1], width=0.0001)
# plt.show(dpi=1500)
# plt.savefig("biased.pdf")
