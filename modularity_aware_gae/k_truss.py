import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

# Function to calculate the support of each edge
def calculate_support(G):
    support = {}
    for edge in G.edges():
        u, v = edge
        common_neighbors = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
        support[edge] = common_neighbors
        support[(v, u)] = common_neighbors  # Since the graph is undirected
    return support

# Function to find the K-truss of the graph
def find_k_truss(G, k):
    # Calculate initial support for each edge
    support = calculate_support(G)
    # Filter out edges with support less than k-2
    to_remove = [edge for edge, sup in support.items() if sup < k - 2]
    while to_remove:
        G.remove_edges_from(to_remove)
        support = calculate_support(G)
        to_remove = [edge for edge, sup in support.items() if sup < k - 2]
    return G

# Function to calculate the similarity matrix based on the K-truss
def calculate_similarity_matrix(A, G_k):
    # Initialize the similarity matrix with zeros
    n = A.shape[0]
    # Assuming `n` is the number of nodes
    n = A.shape[0]  # or some other value that gives the number of nodes
    X = np.zeros((n, n))  # This will create an n x n two-dimensional array

    # X = np.zeros_like(A)
    support = calculate_support(G_k)
    sup_max = max(support.values()) if support else 0

    for i in range(n):
        for j in range(n):
            if (i, j) in support:
                sim_eij = (A[i, j] * (support[(i, j)] + 1)) / (sup_max + 1)
                X[i, j] = sim_eij / n
    # Normalize the similarity matrix
    X = X / (1 + np.max(X))
    return X

# # Assuming A is your adjacency matrix as a numpy array
# # A = np.array([...])  # Replace with your actual adjacency matrix
#
# num_nodes = 10
# A = np.zeros((num_nodes, num_nodes))
#
# # Add edges based on the network topology.
# # This is an undirected graph, so we need to add both A[i][j] and A[j][i].
# edges = [
#     (1, 2), (1, 4), (1, 8),
#     (2, 3), (2, 4), (2, 5),
#     (3, 5),
#     (4, 5), (4, 7), (4, 8),
#     (5, 6), (5, 7),
#     (6, 7),
#     (7, 8),
#     (8, 9),
#     (9, 10)
# ]
#
# # Fill in the adjacency matrix for each edge
# for edge in edges:
#     i, j = edge
#     A[i-1][j-1] = 1  # Subtract 1 because nodes are 1-indexed but Python arrays are 0-indexed
#     A[j-1][i-1] = 1  # Add the edge in both directions since this is an undirected graph
#
# # Now A is the adjacency matrix representing the network topology
# print(A)
#
# # Convert adjacency matrix to graph
# G = nx.from_numpy_matrix(A)
#
# # Apply the K-truss algorithm to find the K-truss subgraph
# k = 4  # Starting with k=3, but you can adjust this based on your needs
# G_k = find_k_truss(G, k)
#
# # Convert the K-truss subgraph back to an adjacency matrix
# A_k = nx.to_numpy_array(G_k)
#
#
#
# # Calculate the similarity matrix X
# X = calculate_similarity_matrix(A, G_k)
#
# # X is the similarity matrix
# print(X)
