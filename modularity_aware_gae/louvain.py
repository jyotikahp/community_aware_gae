import community as cm
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import igraph as ig
import leidenalg

flags = tf.app.flags
FLAGS = flags.FLAGS

def leiden_clustering( adj, s_rec):
    # Convert networkx graph to igraph
    graph = nx.from_scipy_sparse_matrix(adj)
    ig_graph = ig.Graph.from_networkx(graph)

    # Apply the Leiden algorithm
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

    # Print the community of each node
    # for node in range(len(ig_graph.vs)):
    #     print(f"Node {node} is in community {partition.membership[node]}")

    communities_leiden = partition.membership

    # Number of communities found by the Louvain method
    nb_communities_leiden = np.max(communities_leiden) + 1

    # One-hot representation of communities
    communities_leiden_onehot = sp.csr_matrix(np.eye(nb_communities_leiden)[communities_leiden])

    # Community membership matrix (adj_louvain[i,j] = 1 if nodes i and j are in the same community)
    adj_leiden = communities_leiden_onehot.dot(communities_leiden_onehot.transpose())

    # Remove the diagonal
    adj_leiden = adj_leiden - sp.eye(adj_leiden.shape[0])

    # s-regular sparsification of adj_leiden
    adj_leiden = sparsification(adj_leiden, s_rec)

    return adj_leiden, nb_communities_leiden, {}





    ig_graph = ig.Graph.from_networkx(graph)

    # Apply the Leiden algorithm
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)

    # Print the community of each node
    # for node in range(len(ig_graph.vs)):
    #     print(f"Node {node} is in community {partition.membership[node]}")

    return partition.membership


def louvain_clustering(adj, s_rec):

    """
    Performs community detection on a graph using the Louvain method
    :param adj: sparse adjacency matrix of the graph
    :param s_rec: s hyperparameter for s-regular sparsification
    :return: adj_louvain, the Louvain community membership matrix obtained;
    nb_communities_louvain, the number of communities; partition, the community
    associated with each node from the graph
    """

    # Community detection using the Louvain method
    partition = cm.best_partition(nx.from_scipy_sparse_matrix(adj))
    communities_louvain = list(partition.values())

    # communities_louvain = apply_leiden_algorithm(nx.from_scipy_sparse_matrix(adj))

    # Number of communities found by the Louvain method
    nb_communities_louvain = np.max(communities_louvain) + 1

    # One-hot representation of communities
    communities_louvain_onehot = sp.csr_matrix(np.eye(nb_communities_louvain)[communities_louvain])

    # Community membership matrix (adj_louvain[i,j] = 1 if nodes i and j are in the same community)
    adj_louvain = communities_louvain_onehot.dot(communities_louvain_onehot.transpose())

    # Remove the diagonal
    adj_louvain = adj_louvain - sp.eye(adj_louvain.shape[0])

    # s-regular sparsification of adj_louvain
    adj_louvain = sparsification(adj_louvain, s_rec)

    return adj_louvain, nb_communities_louvain, {}


def sparsification(adj_louvain, s = 1):

    """
    Performs an s-regular sparsification of the adj_louvain matrix (if possible)
    :param adj_louvain: the initial community membership matrix
    :param s: value of s for s-regular sparsification
    :return: s-sparsified adj_louvain matrix
    """

    # Number of nodes
    n = adj_louvain.shape[0]

    # Compute degrees
    degrees = np.sum(adj_louvain, axis = 0).getA1()

    for i in range(n):

        # Get non-null neighbors of i
        edges = sp.find(adj_louvain[i,:])[1]

        # More than s neighbors? Subsample among those with degree > s
        if len(edges) > s:
            # Neighbors of i with degree > s
            high_degrees = np.where(degrees > s)
            edges_s = np.intersect1d(edges, high_degrees)
            # Keep s of them (if possible), randomly selected
            removed_edges = np.random.choice(edges_s, min(len(edges_s), len(edges) - s), replace = False)
            adj_louvain[i, removed_edges] = 0.0
            adj_louvain[removed_edges, i] = 0.0
            degrees[i] = s
            degrees[removed_edges] -= 1

    adj_louvain.eliminate_zeros()

    return adj_louvain
