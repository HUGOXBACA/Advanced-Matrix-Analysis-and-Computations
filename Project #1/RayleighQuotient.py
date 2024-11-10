import numpy as np
import matplotlib.pyplot as plt

# Function to compute Rayleigh quotient
def rayleigh_quotient(A, x):
    return (x.T @ A @ x) / (x.T @ x)

# Define mesh grid for visualization
x_vals = np.linspace(-1, 1, 100)
y_vals = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Case (a): Symmetric matrix with two different eigenvalues
A_symmetric = np.array([[2, 1], [1, 3]])
eigenvalues_sym, eigenvectors_sym = np.linalg.eig(A_symmetric)

# Case (b): Non-symmetric matrix with two real eigenvalues
A_non_symmetric = np.array([[2, 3], [0, 1]])
eigenvalues_non_sym, eigenvectors_non_sym = np.linalg.eig(A_non_symmetric)

# Function to plot Rayleigh quotient surface with eigenvectors
def plot_rayleigh_surface_with_eigenvectors(A, eigenvalues, eigenvectors, title):
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])
            Z[i, j] = rayleigh_quotient(A, x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Rayleigh Quotient')

    # Overlay eigenvectors
    for i in range(eigenvectors.shape[1]):
        eigvec = eigenvectors[:, i]
        eigvec_norm = eigvec / np.linalg.norm(eigvec)  # Normalize eigenvector for better visualization
        ax.quiver(0, 0, 0, eigvec_norm[0], eigvec_norm[1], rayleigh_quotient(A, eigvec), color='r', arrow_length_ratio=0.1)
        ax.text(eigvec_norm[0], eigvec_norm[1], rayleigh_quotient(A, eigvec), f'λ{i+1}={eigenvalues[i]:.2f}', color='red')

    plt.show()

# Plot for symmetric matrix with two different eigenvalues
plot_rayleigh_surface_with_eigenvectors(A_symmetric, eigenvalues_sym, eigenvectors_sym, 'Rayleigh Quotient Surface (Symmetric A) with Eigenvectors')

# Plot for non-symmetric matrix with two real eigenvalues
plot_rayleigh_surface_with_eigenvectors(A_non_symmetric, eigenvalues_non_sym, eigenvectors_non_sym, 'Rayleigh Quotient Surface (Non-Symmetric A) with Eigenvectors')
