import csv
import networkx as nx


def get_nx_graph(path):
    G = nx.Graph()
    with open(path) as graph_file:
        graph_csv = csv.reader(graph_file, delimiter=' ')
        for row in graph_csv:
            G.add_edge(int(row[0]), int(row[1]))
    return G