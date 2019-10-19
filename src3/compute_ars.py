from sklearn import cluster

from distance import calc_ars, get_gmm_clusters, principal_angle_distance
import click
from plot_utils import get_nx_graph
from utils import get_as_numpy_array, map_embeddings_to_consecutive





@click.command()
@click.option("-e1", "--embeddings_1", type=str, required=True)
@click.option("-e2", "--embeddings_2", type=str, required=True)
@click.option("-g", "--graph_path", type=str)
@click.option("-c", "--clusters", type=int, required=True)
@click.option("-m", "--method", type=click.Choice(['kmeans', 'gmm']))
def ars(embeddings_1, embeddings_2, graph_path, clusters, method):
    emb1, emb2 = map_embeddings_to_consecutive([embeddings_1, embeddings_2])
    if method == 'kmeans':
        prediction1 = cluster.KMeans(n_clusters=clusters, random_state=0).fit(emb1).labels_
        prediction2 = cluster.KMeans(n_clusters=clusters, random_state=0).fit(emb2).labels_
    else:
        prediction1 = get_gmm_clusters(emb1, clusters)
        prediction2 = get_gmm_clusters(emb2, clusters)
    click.echo(f"Adjusted Rand Score: {calc_ars(prediction1, prediction2)}")


if __name__ == '__main__':
    ars()
