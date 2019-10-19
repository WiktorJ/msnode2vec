import numpy as np
import csv


def get_as_numpy_array(file_path):
    with open(file_path) as file:
        features = csv.reader(file, delimiter=' ')
        header = next(features, None)
        feature_array = np.zeros((int(header[0]), int(header[1])))
        for feature in features:
            feature_array[int(feature[0]) - 1] = feature[1:]
        return np.array(feature_array, dtype=float)



def map_embeddings_to_consecutive(embs):
    mapping = {}
    arrays = []
    with open(embs[0]) as file:
        features = csv.reader(file, delimiter=' ')
        header = next(features, None)
        feature_array = np.zeros((int(header[0]), int(header[1])))
        for i, feature in enumerate(features):
            mapping[int(feature[0])] = i
            feature_array[i] = feature[1:]

    arrays.append(np.array(feature_array, dtype=float))

    for emb in embs[1:]:
        with open(emb) as file:
            features = csv.reader(file, delimiter=' ')
            header = next(features, None)
            feature_array = np.zeros((int(header[0]), int(header[1])))
            for i, feature in enumerate(features):
                feature_array[mapping[int(feature[0])]] = feature[1:]

        arrays.append(np.array(feature_array, dtype=float))
    return arrays


def adjacency_matrix_to_edgelist(matrix):
    edgelist = []
    for i, row in enumerate(matrix):
        for j, el in enumerate(row):
            if el != 0:
                edgelist.append((i + 1, j + 1))
    return edgelist


def reduce_if_exceeds(tuple, treshold=0, reduction=1):
    a, b = tuple
    if a > treshold:
        a -= reduction
    if b > treshold:
        b -= reduction
    return a, b


def find_index_smaller_larger_than(o, array):
    for i, el in enumerate(sorted(array)):
        if el > o:
            return i
    return len(array)


def get_loops(array):
    res = []
    looped = set()
    loops = []
    nodes = set()
    for a, b in array:
        if a == b:
            looped.add(int(a))
            loops.append((int(a), int(b)))
        else:
            res.append((int(a), int(b)))
            nodes.add(int(a))
            nodes.add(int(b))
    print(f"nr of looped {len(looped)}, nr of nodes{len(nodes)}, only looped: {len(looped - nodes)}")
    print(f"edges: {len(array)}")
    return sorted(looped - nodes), loops, res


def delete_looped_labels(edgelist, labels):
    loops, _, _ = get_loops(edgelist)
    return [(int(el[0]) - find_index_smaller_larger_than(int(el[0]), loops), int(el[1])) for el in labels if
            int(el[0]) not in loops]


def delete_loops(array):
    only_looped, loops, res = get_loops(array)
    for loop in loops:
        if loop[0] not in only_looped:
            res.append(loop)
    print(f"after removing loops: {len(res)}")

    return [(a - find_index_smaller_larger_than(a, only_looped),
             b - find_index_smaller_larger_than(b, only_looped)) for a, b in res],


def get_edgelist_from_file(file_path):
    with open(file_path) as file:
        return [tuple(map(int, el)) for el in csv.reader(file, delimiter='\t')]


def save_edgelist_to_file(file_path, edgelist):
    with open(file_path, 'w') as file:
        csv_writer = csv.writer(file, delimiter=' ')
        csv_writer.writerows(edgelist)


def extract_communities(graph, labels, communities):
    extracted_labels = [el[0] for el in labels if el[1] in communities]
    return [el for el in graph if el[0] in extracted_labels and el[1] in extracted_labels], \
           [label for label in labels if label[1] in communities]


def reduce_node_ids(graph, labels):
    nodes = set(el for tup in graph for el in tup)
    mapping = {}
    for i, node in enumerate(nodes):
        mapping[node] = i
    return [(mapping[el[0]], mapping[el[1]]) for el in graph], [(mapping[label[0]], label[1]) for label in labels if
                                                                mapping.get(label[0]) is not None]


graph = get_edgelist_from_file('../graph/com-youtube.ungraph.txt')
# save_edgelist_to_file('../graph/lesmis-fix.edgelist', [reduce_if_exceeds(el) for el in graph])
# labels = get_edgelist_from_file('../labels/email-Eu-core-department-labels-nl.txt')
# new_graph, new_labels = extract_communities(graph, labels, {2, 3, 5, 8, 9})
# new_graph, new_labels = reduce_node_ids(graph, [])
save_edgelist_to_file('../graph/com-youtube-fix.ungraph.txt', reduce_node_ids(graph, [])[0])
# edgelist = get_edgelist_from_file('../graph/email-Eu-core-nl.edgelist')
# save_edgelist_to_file('../graph/email-Eu-core-small.edgelist', extract_communities(edgelist, labels, {2, 3, 5, 8, 9}))
# labels_d = delete_looped_labels(edgelist, labels)
# save_edgelist_to_file('../labels/email-Eu-core-department-labels-nl.txt', labels_d)
# with open('../graph/email-Eu-core.txt') as file:
#     edge_list = csv.reader(file, delimiter=' ')
#     # np_matrix = np.array(list(matrix), dtype=int)
#     # edgelist = adjacency_matrix_to_edgelist(np_matrix)
#     no_loops = delete_loops(list(edge_list))
#
# with open('../graph/email-Eu-core-nl.edgelist', 'w') as file:
#     csv_writer = csv.writer(file, delimiter=' ')
#     csv_writer.writerows(no_loops)

# with open('../graph/lesmis.matrix') as file:
#     matrix = csv.reader(file, delimiter=' ')
#     np_matrix = np.array(list(matrix), dtype=int)

# edgelist = [reduce_if_exceeds(t) for t in edgelist]
# edgelist = [reduce_if_exceeds(t, treshold=45) for t in edgelist]
# with open('../graph/lesmis.edgelist', 'w') as file:
#     csv_writer = csv.writer(file, delimiter=' ')
#     csv_writer.writerows(edgelist)
