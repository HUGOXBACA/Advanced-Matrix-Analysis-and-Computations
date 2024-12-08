import numpy as np
import matplotlib.pyplot as plt

def Solution(A):
    # Perform Singular Value Decomposition
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T

    # Extract singular values and form scaled singular vectors
    u1 = U[:, 0] * s[0]
    v1 = V[:, 0]
    u2 = U[:, 1] * s[1]
    v2 = V[:, 1]

    # Re-form U from the scaled singular vectors u1, u2
    U_scaled = np.column_stack((u1, u2))

    # Prepare angle array for plotting
    theta = np.linspace(0, 2 * np.pi, 501)

    # ----- Plot the unit circle and right singular vectors (S) -----
    fig1 = plt.figure(figsize=(10, 5))

    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks([-1, 1])
    ax1.set_yticks([-1, 1])
    # Move axes to origin
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Unit circle
    x = np.cos(theta)
    y = np.sin(theta)
    ax1.plot(x, y, label='S')

    # Right singular vectors
    ax1.plot([0, v1[0]], [0, v1[1]], label='v1')
    ax1.plot([0, v2[0]], [0, v2[1]], label='v2')
    ax1.text(v1[0]/1.5, v1[1]/3, "v1", color='black', fontsize=14)
    ax1.text(v2[0]/1.5, v2[1]/3, "v2", color='black', fontsize=14)
    ax1.set_title('S', fontweight='bold')
    ax1.legend()

    # ----- Plot the ellipse (AS) and left singular vectors -----
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.set_aspect('equal', adjustable='box')
    # Major and minor axes selection based on singular values
    if s[0] >= s[1]:
        ax_major = U_scaled[0, 0]; ay_major = U_scaled[1, 0]
        bx_minor = U_scaled[0, 1]; by_minor = U_scaled[1, 1]
    else:
        ax_major = U_scaled[0, 1]; ay_major = U_scaled[1, 1]
        bx_minor = U_scaled[0, 0]; by_minor = U_scaled[1, 0]

    # Parametric equation of the ellipse
    x_ell = ax_major * np.sin(theta) + bx_minor * np.cos(theta)
    y_ell = ay_major * np.sin(theta) + by_minor * np.cos(theta)
    ax2.plot(x_ell, y_ell, label='AS')

    # Left singular vectors
    ax2.plot([0, u1[0]], [0, u1[1]], label='u1')
    ax2.plot([0, u2[0]], [0, u2[1]], label='u2')
    ax2.text(u1[0]/1.5, u1[1]/3, "u1", color='black', fontsize=14)
    ax2.text(u2[0]/1.5, u2[1]/3, "u2", color='black', fontsize=14)

    # Set ticks for the second plot
    max_s = np.max(s)
    # Create a symmetric range of ticks
    ticks = np.arange(-int(np.ceil(max_s)), int(np.ceil(max_s)) + 1)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    # Move axes to origin
    ax2.spines['left'].set_position('zero')
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title('AS', fontweight='bold')
    ax2.legend()

    # Annotation: we now use ax1.annotate with figure fraction coordinates
    ax1.annotate('A', xy=(0.6, 0.55), xytext=(0.5, 0.55),
                 textcoords='figure fraction', xycoords='figure fraction',
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return True



if __name__ == '__main__':
    # Example 2x2 matrix for testing
    A = np.array([[3, 1],
                  [1, 3]])
    
    # Call the Solution function to visualize the SVD transformation
    Solution(A)
