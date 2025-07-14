import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Step 1: Define nodes and source term
nodes = np.array([...])  # Node coordinates
f_nodes = np.array([...])  # Source term values at nodes (if known)

# Step 2: Create triangular mesh
tri = Delaunay(nodes)

# Step 3: Define local basis functions and derivatives
def linear_basis(node_coords, x, y):
    # Implement linear basis function evaluation for given node coordinates and (x, y) point
    pass

def grad_linear_basis(node_coords, x, y):
    # Implement gradient of linear basis function evaluation for given node coordinates and (x, y) point
    pass

# Step 4: Compute local matrices and vectors
local_matrices = []
local_vectors = []

for simplex in tri.simplices:
    nodes_in_simplex = nodes[simplex]

    # Compute local stiffness matrix K_e and load vector F_e
    K_e = np.zeros((3, 3))
    F_e = np.zeros(3)

    for i in range(3):
        for j in range(3):
            K_e[i, j] = integrate_over_triangle(linear_basis(nodes_in_simplex[i], x, y) * 
                                                grad_linear_basis(nodes_in_simplex[j], x, y))

        F_e[i] = integrate_over_triangle(f_nodes[simplex[i]] * 
                                         linear_basis(nodes_in_simplex[i], x, y))

    local_matrices.append(K_e)
    local_vectors.append(F_e)

# Step 5: Assemble global matrix and vector
K = csr_matrix(np.block(local_matrices))
F = np.concatenate(local_vectors)

# Step 6: Solve linear system
U = spsolve(K, F)

# Step 7: Postprocessing
def interpolate_solution(tri, nodes, U, x, y):
    # Implement interpolation function to compute solution u(x, y) based on node values U and the triangulation
    pass

u_interpolated = interpolate_solution(tri, nodes, U, x, y)