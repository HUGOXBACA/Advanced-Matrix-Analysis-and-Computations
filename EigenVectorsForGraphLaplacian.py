import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Function to compute Laplacian and eigenvectors
def compute_laplacian_eigenvectors(graph):
    L = nx.laplacian_matrix(graph).toarray()  # Compute Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(L)  # Compute eigenvalues and eigenvectors
    return eigenvalues, eigenvectors

# Function to plot graph
def plot_graph(graph, eigenvectors, title, layout='2D'):
    plt.figure()
    if layout == '2D':
        pos = nx.spring_layout(graph)  # 2D layout
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    elif layout == '1D':
        pos = nx.shell_layout(graph)  # 1D layout
        nx.draw(graph, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)

    plt.title(title)
    plt.show()

# Path graph (1D visualization)
G_path = nx.path_graph(10)
eigenvalues_path, eigenvectors_path = compute_laplacian_eigenvectors(G_path)
plot_graph(G_path, eigenvectors_path, 'Path Graph (1D)', layout='1D')

# Circle graph (1D and 2D visualization)
G_circle = nx.cycle_graph(10)
eigenvalues_circle, eigenvectors_circle = compute_laplacian_eigenvectors(G_circle)

# 1D visualization for Circle Graph
plot_graph(G_circle, eigenvectors_circle, 'Circle Graph (1D)', layout='1D')

# 2D visualization for Circle Graph
plot_graph(G_circle, eigenvectors_circle, 'Circle Graph (2D)', layout='2D')

# Cube graph (1D and 2D visualization)
G_cube = nx.grid_graph(dim=[2, 2, 2])  # Cube graph in 3D
eigenvalues_cube, eigenvectors_cube = compute_laplacian_eigenvectors(G_cube)

# 1D visualization for Cube Graph
plot_graph(G_cube, eigenvectors_cube, 'Cube Graph (1D)', layout='1D')

# 2D visualization for Cube Graph
plot_graph(G_cube, eigenvectors_cube, 'Cube Graph (2D)', layout='2D')
