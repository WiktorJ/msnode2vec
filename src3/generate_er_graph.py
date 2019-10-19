from networkx import nx

for p in range(10, 18):
    n = 2 ** p  # nodes
    m = n * 20  # edges

    G = nx.gnm_random_graph(n, m)
    nx.write_edgelist(G, f"../graph/er_graph_{n}", data=False)

