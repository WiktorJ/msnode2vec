import numpy as np
import random
from itertools import islice, chain


class WalkAggregator:

    def __init__(self, current: int, walks: list, walks_length: int = 1, prev: int = None):
        self.current = current
        self.prev = prev
        self.walks_length = walks_length
        self.walks_count = len(walks)
        self.walks = walks

    def inc_len(self):
        self.walks_length += 1

    def __hash__(self) -> int:
        return hash(self.current) if self.prev is None else hash((self.current, self.prev))

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and self.current == other.current and self.prev == other.prev


class Graph:

    def __init__(self, nx_G, is_directed, p, q, log_stats=False):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.neighbors = {}

    def chunker(self, iterable, n):
        it = iter(iterable)
        while True:
            chunk = islice(it, n)
            try:
                first = next(chunk)
            except StopIteration:
                return
            yield chain((first,), chunk)

    def draw_node(self, node, node_neighbors):
        n1 = self.alias_nodes[node][0]
        n2 = self.alias_nodes[node][1]
        alias = alias_draw(n1, n2)
        return node_neighbors[alias]

    def draw_edge(self, curr, prev, node_neighbors):
        n1, n2 = self.alias_edges[(prev, curr)]
        alias = alias_draw(n1, n2)
        return node_neighbors[alias]

    def update_step(self, current_node, walks, visit_dict_next, completed_walks, walk_len):
        # np.random.shuffle(drawn) consider if this is necessary (or some other way of randomizing the order).
        # Simply iterating drawn might introduce some bias

        for walk in walks:
            if len(walk) == 1:
                new_node = self.draw_node(current_node, self.neighbors[current_node])
            else:
                new_node = self.draw_edge(current_node, walk[-2], self.neighbors[current_node])
            new_walk = walk + [new_node]
            if len(new_walk) == walk_len:
                completed_walks.append(new_walk)
            else:
                visit_dict_next[new_node] = visit_dict_next.get(new_node, [])
                visit_dict_next[new_node].append(new_walk)

    def get_prev(self, walk):
        return walk[-2] if len(walk) > 1 else -1

    def node2vec_walk(self, walk_length, start_nodes, num_walks):
        '''
        Simulate a random walk starting from start node.
        '''

        visit_dict = {}
        visit_dict_next = {}
        completed_walks = []
        G = self.G

        for start_node in start_nodes:
            visit_dict[start_node] = [[start_node] for _ in range(num_walks)]

        while visit_dict:
            while visit_dict:
                current_node, walks = visit_dict.popitem()
                cur_nbrs = self.neighbors[current_node]

                if len(cur_nbrs) > 0:
                    if len(walks[0]) == 1:
                        # drawn = self.draw_node(current_node, le/n(walks), cur_nbrs)
                        self.update_step(current_node, walks, visit_dict_next, completed_walks, walk_length)
                    else:
                        # drawn = self.draw_edge(current_node, len(walks), cur_nbrs)
                        self.update_step(current_node, walks, visit_dict_next, completed_walks, walk_length)
                else:
                    print("HANDLE NODE WITH NO NEIGHBORS")  # Probably just continue
            visit_dict, visit_dict_next = visit_dict_next, visit_dict

        return completed_walks

    def simulate_walks(self, num_walks, walk_length, concurrent_nodes=16):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for node in nodes:
            self.neighbors[node] = sorted(G.neighbors(node))
        random.shuffle(nodes)
        i = 0
        for node_chunk in self.chunker(nodes, concurrent_nodes):
            # i += 1
            # if i % 100 == 0:
            #     print(
            #         f"processed {i} chunks of {int(len(nodes) / concurrent_nodes)} in total. Chunk size: {concurrent_nodes}")
            walks += self.node2vec_walk(walk_length=walk_length, start_nodes=list(node_chunk), num_walks=num_walks)
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.random.rand() * K)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
