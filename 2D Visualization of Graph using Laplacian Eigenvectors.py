import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a connected graph
# For demonstration, let's use a simple connected graph like a star graph
G = nx.star_graph(5)  # Star graph with a central node and 5 leaves

# Step 2: Calculate the Laplacian matrix of the graph
L = nx.laplacian_matrix(G).toarray()

# Step 3: Compute the eigenvalues and eigenvectors of the Laplacian matrix
eigenvalues, eigenvectors = np.linalg.eigh(L)  # eigh for symmetric (Hermitian) matrices

# Step 4: Use the first two non-trivial eigenvectors for 2D positioning
# The first eigenvector corresponds to the zero eigenvalue and is typically ignored
pos = {i: (eigenvectors[i, 1], eigenvectors[i, 2]) for i in range(len(G.nodes()))}

# Step 5: Plot the graph using the computed positions
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=12)
plt.title("2D Visualization of Graph using Laplacian Eigenvectors")
plt.show()
