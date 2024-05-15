import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt

"""
Disclaimer: the parse_index_file function from this file, as well as the
cora/citeseer/pubmed parts of the loading functions, come from the
tkipf/gae original repository on Graph Autoencoders
"""


def parse_index_file(filename):

    index = []
    for line in open(filename):
        index.append(int(line.strip()))

    return index


def load_data(dataset, labels):

    """
    Load datasets
    :param dataset: name of the input graph dataset
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """

    if dataset == 'cora-large':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/coralarge", delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset == 'sbm':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/sbm.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'blogs':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/karate.edgelist",
                                                   nodetype = int,
                                                   data = (('weight', int),),
                                                   delimiter = ','))
        graph = nx.from_scipy_sparse_matrix(adj)

        features = sp.identity(adj.shape[0])


    elif dataset in ('cora', 'citeseer', 'pubmed'):
        # Load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding = 'latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(graph)
        # print("Adjacency matrix ", adj)

        # draw_graph(graph, labels)

        # Function to convert an adjacency matrix to a NetworkX graph
        def adjacency_matrix_to_graph(adj_matrix):
            return nx.from_scipy_sparse_matrix(adj_matrix)

        # Example usage:
        # Let's assume `adj_matrix` is your adjacency matrix.
        # Replace `adj_matrix` with your actual adjacency matrix variable.
        # adj_matrix = ...

        # Convert the adjacency matrix to a NetworkX graph
        # G = adjacency_matrix_to_graph(adj_matrix=adj)

        # Create a list or dictionary to hold groups and their nodes
        # groups_with_nodes = {}
        #
        # for i, component in enumerate(connected_components):
        #     # Assign all nodes in the component to the group
        #     groups_with_nodes[f"Group {i}"] = list(component)
        # Note: Make sure to replace `adj_matrix` with your actual adjacency matrix.

        # print(groups_with_nodes)

    else:
        raise ValueError('Error: undefined dataset!')

    return adj, features, graph


def load_labels(dataset):

    """
    Load node-level labels
    :param dataset: name of the input graph dataset
    :return: n-dim array of node labels, used for community detection
    """

    if dataset == 'cora-large':
        labels = np.loadtxt("../data/coralarge-cluster", delimiter = ' ', dtype = str)

    elif dataset == 'sbm':
        labels = np.repeat(range(100), 1000)

    elif dataset == 'blogs':
        labels = np.loadtxt("../data/karate.labels", delimiter = ' ', dtype = str)

    elif dataset in ('cora', 'citeseer', 'pubmed'):
        names = ['ty', 'ally']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding = 'latin1'))
                else:
                    objects.append(pkl.load(f))
        ty, ally = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        labels = sp.vstack((ally, ty)).tolil()
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # One-hot to integers
        labels = np.argmax(labels.toarray(), axis = 1)

    else:
        raise ValueError('Error: undefined dataset!')

    return labels

def draw_graph(graph, labels, FLAGS):
    G = graph

    # Convert adjacency matrix to graph
    # G = nx.from_scipy_sparse_matrix(adj_init)  # assuming adj_init is a numpy array
    # G = nx.from_numpy_array(adj_matrix)

    # labels is a list of community labels corresponding to each node index
    # We will create a dictionary from it to use with networkx
    community_labels = {i: label for i, label in enumerate(labels)}

    # Assign the community labels to each node in the graph
    nx.set_node_attributes(G, community_labels, 'community')

    # Choose a layout for your graph
    # pos = nx.spring_layout(G, seed=42)  # Force consistency in the layout
    pos = nx.kamada_kawai_layout(G)

    # Draw the graph according to community
    # Get the unique communities and assign a color to each
    unique_communities = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
    community_color = dict(zip(unique_communities, colors))

    # Draw nodes with community colors
    for community in unique_communities:
        nodes_of_community = [node for node, attr in G.nodes(data=True) if attr['community'] == community]
        nx.draw_networkx_nodes(G, pos, nodes_of_community, label=str(community),
                               node_size=50, node_color=[community_color[community]])

    # Optionally, draw the edges. You can make them semi-transparent or skip drawing them to reduce visual clutter.
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

    # Disable axis as they are not meaningful for this kind of plot
    plt.axis('off')

    # Add a legend if you want to indicate what community each color represents
    # plt.legend(scatterpoints=1, title='Communities')
    plt.savefig("Communities for "+ FLAGS.dataset+" run "+FLAGS.pre_processing_algo+ str(FLAGS.features)+FLAGS.model)

    # Show the graph
    # plt.show()
