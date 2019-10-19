#!/usr/bin/python3
import networkx as nx
import node2vec_ms as node2vec
from gensim.models import Word2Vec
from line_profiler import LineProfiler

config = {
    # 'input': '../graph/facebook_combined.edgelist',
    'input': '../graph/lesmis.edgelist',
    # 'output': '../emb/lesmis{}.emb',
    'dimensions': 16,
    'walk_length': 80,
    'num_walks': 120,
    'window_size': 10,
    'iter': 10,
    'workers': 8,
    'p': 1,
    'q': 0.5,
    'weighted': False,
    'directed': False,
    'unweighted': True,
    'concurrent_walks': 128

}


def read_graph(config):
    '''
    Reads the input network in networkx.
    '''
    if config['weighted']:
        G = nx.read_edgelist(config['input'], nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(config['input'], nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not config['directed']:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, config):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=config['dimensions'], window=config['window_size'], min_count=0, sg=1,
                     workers=config['workers'], iter=config['iter'])

    model.wv.save_word2vec_format(config['output'])

    return list(zip(model.wv.vectors, model.wv.index2entity))


def main(config):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''

    nx_G = read_graph(config)
    G = node2vec.Graph(nx_G, config['directed'], config['p'], config['q'])
    G.preprocess_transition_probs()
    walks = G.simulate_walks(config['num_walks'], config['walk_length'], config['concurrent_walks'])
    # learn_embeddings(walks, config)


# learn_embeddings(walks)

if __name__ == "__main__":
    lp = LineProfiler()
    lp.add_function(node2vec.Graph.chunker)
    lp.add_function(node2vec.Graph.draw_node)
    lp.add_function(node2vec.Graph.draw_edge)
    lp.add_function(node2vec.Graph.update_step)
    lp.add_function(node2vec.Graph.get_prev)
    lp.add_function(node2vec.Graph.node2vec_walk)
    lp.add_function(node2vec.Graph.simulate_walks)
    lp.add_function(node2vec.Graph.get_alias_edge)
    lp.add_function(node2vec.Graph.preprocess_transition_probs)
    lp.add_function(node2vec.alias_draw)
    lp.add_function(node2vec.alias_setup)
    lp.add_function(learn_embeddings)
    lp.add_function(read_graph)
    lp_wrapper = lp(main)
    lp_wrapper(config)
    lp.print_stats()
