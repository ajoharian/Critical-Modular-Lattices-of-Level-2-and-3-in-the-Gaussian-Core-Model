import os
import pickle
import numpy as np
import lattice_utils as gu
from scipy.linalg import eigh


class Lattice:

    def __init__(self, name, vectors, real_vectors=None, dimension=None, eigenvalues=None, eigenvectors=None,
                 grammatrix=None, shell_design=None, quadratic_form_per_shell=None, eigenvalue_groups=None,
                 coefficients_theta_series=None):
        self.name = name
        self.vectors = vectors
        self.real_vectors = real_vectors if real_vectors is not None else {}
        self.dimension = dimension if dimension is not None else -1
        self.eigenvalues = eigenvalues if eigenvalues is not None else {}
        self.eigenvectors = eigenvectors if eigenvectors is not None else {}
        # TODO ist np.empty(0) hier richtig?
        self.grammatrix = grammatrix if grammatrix is not None else np.empty(0)
        self.shell_design = shell_design if shell_design is not None else -1
        self.quadratic_form_per_shell = quadratic_form_per_shell if quadratic_form_per_shell is not None else {}
        self.eigenvalue_groups = eigenvalue_groups if eigenvalue_groups is not None else []
        self.coefficients_theta_series = coefficients_theta_series if coefficients_theta_series is not None else []

    def save_as_pickle(self, directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

            # Construct the full file path
        path = os.path.join(directory_path, self.name + '.pkl')

        # Save the object as a pickle file
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def get_cholesky_decomposition(self):
        # Check if inner_product_matrix is symmetric
        if not np.allclose(self.grammatrix, self.grammatrix.T):
            raise ValueError("Matrix inner_product_matrix must be symmetric.")

        # Check if inner_product_matrix is positive definite
        if not np.all(np.linalg.eigvals(self.grammatrix) > 0):
            raise ValueError("Matrix inner_product_matrix must be positive definite.")

        # Compute Cholesky decomposition
        cholesky_decomp = np.linalg.cholesky(self.grammatrix)

        return cholesky_decomp

    def compute_real_vectors(self):
        real_vectors = {}
        cholesky_matrix = self.get_cholesky_decomposition()
        for length, vector_list in self.vectors.items():
            real_vector_list = []
            for vector in vector_list:
                real_vec = vector @ cholesky_matrix
                minus_real_vec = - vector @ cholesky_matrix
                real_vector_list.append(real_vec)
                real_vector_list.append(minus_real_vec)
            real_vectors[length] = real_vector_list
        self.real_vectors = real_vectors

    def compute_dimension(self):
        self.dimension = self.grammatrix.shape[0]

    def compute_eigenvalues(self):
        basis_tsm = gu.build_basis_tsm(self.dimension)
        n = len(basis_tsm)
        ev_matrix = np.zeros((n,n))
        vectors = self.real_vectors
        for i in range(n):
            print(i)
            for j in range(n):
                h1 = basis_tsm[i]
                h2 = basis_tsm[j]
                hp = h1 + h2
                hm = h1 - h2
                laufbursche = 0
                for length,vectorlist in vectors.items():
                    for v in vectorlist:
                        laufbursche = laufbursche + 0.25 * (((hp.dot(v)).dot(v)) ** 2 - ((hm.dot(v)).dot(v)) ** 2)
                ev_matrix[i, j] = laufbursche
        w, v = eigh(ev_matrix)
        print(f"Eigenwerte von H für {self.name} {w}")

    def compute_eigenvalues_per_length(self):
        self.eigenvalues = {}
        self.eigenvectors = {}
        basis_tsm = gu.build_basis_tsm(self.dimension)
        n = len(basis_tsm)
        self.quadratic_form_per_shell = {}
        #print(n)
        #print(self.quadratic_form_per_shell)
        ev_matrix = np.zeros((n, n))
        vectors = self.real_vectors
        for length,vectorlist in vectors.items():
            for i in range(n):
                #print(i)
                for j in range(i, n):
                    h1 = basis_tsm[i]
                    h2 = basis_tsm[j]
                    hp = h1 + h2
                    hm = h1 - h2
                    laufbursche = 0
                    for v in vectorlist:
                        laufbursche = laufbursche + 0.25 * (((hp.dot(v)).dot(v)) ** 2 - ((hm.dot(v)).dot(v)) ** 2)
                    ev_matrix[i, j] = laufbursche
                    ev_matrix[j, i] = laufbursche
            eigenvalues, eigenvectors = eigh(ev_matrix)
            #print(f"Eigenwerte von H für {self.name} {eigenvalues}")
            #print(ev_matrix)
            self.eigenvalues[length] = eigenvalues
            self.quadratic_form_per_shell[length] = ev_matrix.copy()
            self.eigenvectors[length] = eigenvectors

    @classmethod
    def load_from_pickle(cls, file_path):
        # Load the object from a pickle file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'rb') as file:
            obj = pickle.load(file)

        return obj


