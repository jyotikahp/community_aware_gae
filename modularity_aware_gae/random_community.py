import numpy as np

def create_community_matrix(n, k):
    # Randomly assign communities to nodes
    communities = np.random.randint(0, k, size=n)
    # Create an empty adjacency matrix
    community_matrix = np.zeros((n, n))
    # Assign 1 to aij if node i and node j belong to the same community
    for i in range(n):
        for j in range(i+1, n):  # Fill only upper triangular part for efficiency
            if communities[i] == communities[j]:
                community_matrix[i, j] = 1
    # Make the matrix symmetric
    community_matrix = community_matrix + community_matrix.T
    return community_matrix

def generate_community_adjacency_matrix(n):
    # Set the number of communities k as up to n/2
    k = max(n // 2, 1)  # At least one community
    return create_community_matrix(n, k)
# Example usage:
n = 8  # Number of nodes as a parameter
k = 3
# community_matrix, communities = create_community_matrix(n, k)

# print(community_matrix, communities       )
