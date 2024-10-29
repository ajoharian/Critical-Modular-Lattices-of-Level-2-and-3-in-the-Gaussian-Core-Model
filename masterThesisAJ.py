import numpy as np
import lattice_utils as gu
from numpy import exp
from numpy import pi
from numpy import sqrt
from lattice import Lattice
import os
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import latticeIO as lio
import modular_forms_coefficients as mfc
from PIL import Image


def compute_all_real_vectors_and_save_lattices(lattices):
    for lattice in lattices:
        lattice.compute_real_vectors()
        lattice.save_as_pickle('Resources/lattices')


# rounds all values of an array to the nearest integer and returns a set, removing duplicates
def round_all_values_set(array):
    values = set()
    for entry in array:
        values.add(round(entry))
    return values


# rounds all values of an array to 4 decimal places and returns a set, removing duplicates
def round_all_values_to_near_set(array):
    values = set()
    for entry in array:
        values.add(round(entry, 4))
    return values


def round_np_array(array, entry):
    return np.round(array, entry)


# Checks if two matrices commute with error 0.1
def commute(matrix1, matrix2, tol=1e-1):
    return np.allclose(np.dot(matrix1, matrix2), np.dot(matrix2, matrix1), atol=tol)


# Checks if three matrices commute pairwise with error 0.1
def pairwise_commute(matrix1, matrix2, matrix3):
    commute_AB = commute(matrix1, matrix2)
    commute_BC = commute(matrix2, matrix3)
    commute_AC = commute(matrix1, matrix3)
    if commute_AB and commute_BC and commute_AC:
        print("The matrices commute pairwise.")
        return True
    else:
        print("The matrices do not commute pairwise.")
        return False


# groups eigenvalues removing rounding errors
def group_close_eigenvalues(eigenvalues, tolerance=1e-3):
    # Sort the eigenvalues
    sorted_eigenvalues = np.sort(eigenvalues)

    # Initialize the group list
    list_of_grouped_eigenvalues = []
    current_group = [sorted_eigenvalues[0]]

    # Loop through sorted eigenvalues
    for i in range(1, len(sorted_eigenvalues)):
        if np.abs(sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]) < tolerance:
            current_group.append(sorted_eigenvalues[i])
        else:
            list_of_grouped_eigenvalues.append(current_group.copy())
            current_group = [sorted_eigenvalues[i]]
    list_of_grouped_eigenvalues.append(current_group.copy())  # Add the last group
    return list_of_grouped_eigenvalues


# gets eigenspaces for eigenvalues
def get_eigenspaces(eigenvalues, eigenvectors, tolerance=1e-3):
    groups = group_close_eigenvalues(eigenvalues, tolerance)
    eigenspaces = {}
    for group in groups:
        indices = [i for i, val in enumerate(eigenvalues) if np.any(np.isclose(val, group, atol=tolerance))]
        eigenspace = eigenvectors[:, indices]
        eigenspaces[tuple(group)] = eigenspace
    return eigenspaces


# Check if the column space of V1 is a subspace of the column space of V2
def is_subspace(V1, V2, tol=1e-3):
    # Project V1 onto the space spanned by V2
    # Compute the projection matrix of V2: P_V2 = V2 * (V2^T * V2)^(-1) * V2^T
    P_V2 = V2 @ np.linalg.pinv(V2)  # Projection matrix for V2
    # Project each vector in V1 onto V2
    projected_V1 = P_V2 @ V1
    # Check if the projection equals V1 (i.e., if V1 is in the column space of V2)
    return np.allclose(projected_V1, V1, atol=tol)


# Check if a matrix is a diagonal matrix with a tolerance factor for rounding errors
def is_diagonal(matrix, tol=1e-4):
    return np.allclose(matrix, np.diag(np.diagonal(matrix)), atol=tol)


# Computes all the eigenvalues of the quadratic form per shells matrix
def get_all_eigenvals2(lattice):
    print(lattice.name)
    A = lattice.quadratic_form_per_shell[4].copy()
    B = lattice.quadratic_form_per_shell[6].copy()
    if commute(A, B):
        eigval_A, eigvec_A = eigh(A)
        B_prime = eigvec_A.T @ B @ eigvec_A
        if is_diagonal(B_prime):
            print("B' and C' are diagonal in the eigenbasis of A.")
            rounded_B_prime = np.round(B_prime, decimals=5)
            diagonal_B = np.diagonal(rounded_B_prime)
            print(f'eigenvalues of A for lattice ' + lattice.name)
            print(eigval_A)
            print(f'eigenvalues of B for lattice ' + lattice.name)
            print(diagonal_B)
            A_strich = round_all_values_set(eigval_A)
            B_strich = round_all_values_set(diagonal_B)
            print(A_strich)
            print(B_strich)
        else:
            print("B' not diagonal in the eigenbasis of A.")
    else:
        print(f'The matrices are not simultaneously diagonalizable ' + lattice.name)


# checks the common eigenspaces for eigenvalues
def test_eigenspaces_for_eigenvalues2(lattice):
    print(lattice.name)
    A = lattice.quadratic_form_per_shell[4].copy()
    B = lattice.quadratic_form_per_shell[6].copy()
    eigval_A, eigvec_A = eigh(A)
    tolerance = 1e-3
    eigenspaces_A = get_eigenspaces(eigval_A, eigvec_A, tolerance)
    eigval_B, eigvec_B = eigh(B)
    eigenspaces_B = get_eigenspaces(eigval_B, eigvec_B, tolerance)
    for group_A, eigenspace_A in eigenspaces_A.items():
        for group_B, eigenspace_B in eigenspaces_B.items():
            if is_subspace(eigenspace_A, eigenspace_B):
                print(f"Is {group_A[0]} a subspace of {group_B[0]}?", is_subspace(eigenspace_A, eigenspace_B))
            if is_subspace(eigenspace_B, eigenspace_A):
                print(f"Is {group_B[0]} a subspace of {group_A[0]}?", is_subspace(eigenspace_B, eigenspace_A))


# computes pairs of corresponding eigenvalues
def computing_corresponding_ev(lattice):
    ev_array = []
    print(lattice.name)
    A = lattice.quadratic_form_per_shell[4].copy()
    B = lattice.quadratic_form_per_shell[6].copy()
    eigval_A, eigvec_A = eigh(A)
    tolerance = 1e-3
    eigenspaces_A = get_eigenspaces(eigval_A, eigvec_A, tolerance)
    eigval_B, eigvec_B = eigh(B)
    eigenspaces_B = get_eigenspaces(eigval_B, eigvec_B, tolerance)
    for group_A, eigenspace_A in eigenspaces_A.items():
        for group_B, eigenspace_B in eigenspaces_B.items():
            if is_subspace(eigenspace_A, eigenspace_B):
                ev_array.append([np.round(group_A[0], decimals=5), np.round(group_B[0], decimals=5)])
    print(ev_array)


# computes the Hessian for shell 4 and 6
def hessian_eigenvalue_shell4_6(n, Theta, Cusp1, Cusp2, c, e1, e2, N):
    lb = 0
    for m in range(0, N):
        lb = lb + (Cusp1[m] * c ** 2 / 2 * (e1 * n * (n + 2) - 32 * Theta[1]) + Cusp2[m] * c ** 2 / 2 * (
                e2 * n * (n + 2) - 72 * Theta[2])) * exp(-c * 2 * m)
        lb = lb + Theta[m] * 2 * c * m * (2 * c * m - (n / 2 + 1)) * exp(-c * 2 * m)
    lb = 1 / (n * (n + 2)) * lb
    return lb


# computes the hessian for the roots
def hessian_eigenvalue_roots(n, theta, cusp1, c, e, N):
    lb = 0
    for m in range(0, N):
        lb = lb + (cusp1[m] * c ** 2 / 2 * (e * n * (n + 2) - 8 * theta[1])) * exp(-c * 2 * m)
        lb = lb + theta[m] * 2 * c * m * (2 * c * m - (n / 2 + 1)) * exp(-c * 2 * m)
    lb = 1 / (n * (n + 2)) * lb
    return lb


# computes the hessian for 4 designs
def hessian_eigenvalue_4_designs(n, theta, c, N):
    lb = 0
    for m in range(0, N):
        lb = lb + theta[m] * 2 * c * m * (2 * c * m - (n / 2 + 1)) * exp(-c * 2 * m)
    lb = 1 / (n * (n + 2)) * lb
    return lb


# plots and saves the Hessian for shells 4 and 6
def save_as_png_shell4_6(eigenvalues, n, Theta, Cusp1, Cusp2, lattice):
    path = 'Resources/eigenvalues_hessian'
    file_name = lattice.name
    if not os.path.exists(path):
        os.makedirs(path)

    # Construct the full file paths
    plot_png_path = os.path.join(path, file_name + '_plot.png')
    legend_png_path = os.path.join(path, file_name + '_legend.png')

    # Create figure and plot without the legend
    fig, ax = plt.subplots(figsize=(10, 6))

    for e in eigenvalues:
        evplotlist = []
        c_values = []  # Store c values for plotting
        eigenvalues_list = []  # Store the corresponding Hessian eigenvalues
        for steps in range(0, 31):
            c = pi / sqrt(3) + steps / 10
            sum_val = hessian_eigenvalue_shell4_6(n, Theta, Cusp1, Cusp2, c, e[0], e[1], 200)
            evplotlist.append((c, sum_val))
            c_values.append(c)
            eigenvalues_list.append(sum_val)
        ax.plot(c_values, eigenvalues_list, label=f'e1 = {e[0]}, e2 = {e[1]}')

    # Customize the plot
    ax.set_title(f'Hessian Eigenvalue Plots for {lattice.name}')
    ax.set_xlabel('c values')
    ax.set_ylabel('Hessian eigenvalue')
    ax.grid(True)

    # Save the plot without the legend
    plt.savefig(plot_png_path, bbox_inches='tight')
    plt.close(fig)

    # Now create the legend as a separate figure
    fig_legend = plt.figure(figsize=(8, 6))
    handles, labels = ax.get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc='center')

    # Save the legend separately
    fig_legend.savefig(legend_png_path, bbox_inches='tight')
    plt.close(fig_legend)


# plots and saves the Hessian for shell 2
def save_as_png_shell2(eigenvalues, lattice):
    n = lattice.dimension
    Theta = mfc.get_theta_coefficients(lattice)
    Cusp1 = mfc.get_cusp_plus_four_coefficients(lattice)
    path = 'Resources/eigenvalues_hessian'
    file_name = lattice.name
    if not os.path.exists(path):
        os.makedirs(path)

    # Construct the full file paths
    plot_png_path = os.path.join(path, file_name + '_plot.png')
    legend_png_path = os.path.join(path, file_name + '_legend.png')

    # Create figure and plot without the legend
    fig, ax = plt.subplots(figsize=(10, 6))

    for e in eigenvalues:
        evplotlist = []
        c_values = []  # Store c values for plotting
        eigenvalues_list = []  # Store the corresponding Hessian eigenvalues
        for steps in range(0, 31):
            c = pi / sqrt(3) + steps / 10
            sum_val = hessian_eigenvalue_roots(n, Theta, Cusp1, c, e, 200)
            evplotlist.append((c, sum_val))
            c_values.append(c)
            eigenvalues_list.append(sum_val)
        ax.plot(c_values, eigenvalues_list, label=f'e1 = {e}')

    # Customize the plot
    ax.set_title(f'Hessian Eigenvalue Plots for {lattice.name}')
    ax.set_xlabel('c values')
    ax.set_ylabel('Hessian eigenvalue')
    ax.grid(True)

    # Save the plot without the legend
    plt.savefig(plot_png_path, bbox_inches='tight')
    plt.close(fig)

    # Now create the legend as a separate figure
    fig_legend = plt.figure(figsize=(8, 6))
    handles, labels = ax.get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc='center')

    # Save the legend separately
    fig_legend.savefig(legend_png_path, bbox_inches='tight')
    plt.close(fig_legend)


# plots and saves the Hessian for 4 designs
def save_as_png_4_designs(lattice):
    Theta = mfc.get_theta_coefficients(lattice)
    n = lattice.dimension
    path = 'Resources/eigenvalues_hessian'
    file_name = lattice.name
    if not os.path.exists(path):
        os.makedirs(path)
    # Construct the full file paths
    plot_png_path = os.path.join(path, file_name + '_plot.png')
    # Create figure and plot without the legend
    fig, ax = plt.subplots(figsize=(10, 6))
    evplotlist = []
    c_values = []  # Store c values for plotting
    eigenvalues_list = []  # Store the corresponding Hessian eigenvalues
    for steps in range(0, 31):
        c = pi / sqrt(3) + steps / 10
        sum_val = hessian_eigenvalue_4_designs(n, Theta, c, 200)
        evplotlist.append((c, sum_val))
        c_values.append(c)
        eigenvalues_list.append(sum_val)
    ax.plot(c_values, eigenvalues_list, label=f'e1 = {0}, e2 = {1}')

    # Customize the plot
    ax.set_title(f'Hessian Eigenvalue Plots for {lattice.name}')
    ax.set_xlabel('alpha values')
    ax.set_ylabel('Hessian eigenvalue')
    ax.grid(True)

    # Save the plot without the legend
    plt.savefig(plot_png_path, bbox_inches='tight')
    plt.close(fig)


# computes the gradient for a matrix H
def gradient_dict_per_h(lattice, c_values):
    vectorgroup = lattice.real_vectors[2].copy()
    matrix_basis = gu.build_basis_tsm(lattice.dimension)

    # Store the results for each matrix h, indexed by its position
    gradients_per_h = {}

    for idx, h in enumerate(matrix_basis):
        gradients = []
        for c in c_values:
            lb = 0
            for vector in vectorgroup:
                lb = lb + np.dot(np.dot(vector.T, h), vector)

            lb = -lb * c * exp(-2 * c)  # Calculate gradient based on c
            gradients.append(lb)  # Store result for this value of c
        gradients_per_h[idx] = gradients  # Store results using the index of h

    return gradients_per_h


# Gives the lattice for Magma
def get_magma_lattice(lattice):
    matrix = lattice.grammatrix.copy()
    dimension = lattice.dimension
    matrix_list = matrix.flatten().tolist()
    matrix_str = ', '.join(map(str, matrix_list))
    output = f"L := LatticeWithGram(MatrixRing(IntegerRing(), {dimension}) ! [{matrix_str}]);"
    return output


# Show the plot of the Hessian
def show_hessian_plot(lattice):
    if lattice.shell_design > 1:
        img = Image.open(f'Resources/eigenvalues_hessian/' + lattice.name + '_plot.png')
        img.show()
    else:
        print('no plot for this lattice')


# Show legend of the plot of the Hessian
def show_hessian_legend(lattice):
    if lattice.shell_design == 3:
        img = Image.open(f'Resources/eigenvalues_hessian/' + lattice.name + '_legend.png')
        img.show()
    else:
        print('no legend for this plot')


if __name__ == "__main__":

    # Example for the lattice L3D16.3

    # get lattice
    lattice = lio.load_lattice('L3D16.3')
    # get list of corresponding eigenvalues of the Gram matrices for the shells 4 and 6
    ev_pairs = lattice.eigenvalue_groups
    print(ev_pairs)
    # get the first 200 coefficients of theta of L
    theta = mfc.get_theta_coefficients(lattice)
    # get the first 200 coefficients of C2 of L
    c2 = mfc.get_cusp_plus_four_coefficients(lattice)[1]
    # get the first 200 coefficients of C3 of L
    c3 = mfc.get_cusp_plus_four_coefficients(lattice)[2]
    # choose the number of coefficients to use
    N = 50
    # choose alpha
    alpha = 3
    # compute the eigenvalue of the Hessian for each pair of eigenvalues of the Gram matrices using the first N coeffs
    for pair in ev_pairs:
        print(f'The eigenvalue of L for the pair {pair[0]}, {pair[1]} is '
              f'{hessian_eigenvalue_shell4_6(lattice.dimension, theta, c2, c3, alpha, pair[0], pair[1], N)}')
    # Open the plot
    show_hessian_plot(lattice)
    # Open the legend of the plot
    show_hessian_legend(lattice)

    # Example for L2D12.3 no critical point
    # get lattice
    lattice2 = lio.load_lattice('L2D12.3')
    # get the vectors of squared length 2
    roots = lattice2.real_vectors[2].copy()
    # create an empty 12x12 matrix
    mat = np.zeros((12, 12))
    # compute the matrix sum_{x \in L(2)} x x^T
    for root in roots:
        rounded_root = np.round(root, decimals=10)
        mat += (rounded_root[:, np.newaxis] @ rounded_root[np.newaxis, :])
    # round the matrix
    rounded_mat = np.round(mat, decimals=2)
    print("Rounded matrix:")
    print(rounded_mat)

    # Example for Coxeter-Todd
    coxeter_todd = lio.load_lattice('L3D12.1')
    #choose alpha
    alpha_ct = 2
    # Compute the Hessian for the first
    print(f'The Hessian of the Coxeter Todd lattice for alpha = {alpha_ct} for the first 15 coefficients is '
          f'{hessian_eigenvalue_4_designs(coxeter_todd.dimension, mfc.get_theta_coefficients(coxeter_todd),
                                          alpha_ct, 15)}')
