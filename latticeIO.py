import os
from lattice import Lattice
from gitter import Gitter
import concurrent.futures


def get_all_lattices_from(path):
    lattices = []
    pkl_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
    for lattice_path in pkl_files:
        lattices.append(Lattice.load_from_pickle(lattice_path))
    return lattices

def get_all_gitter_from(path):
    lattices = []
    pkl_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
    for lattice_path in pkl_files:
        lattices.append(Gitter.load_from_pickle(lattice_path))
    return lattices


def get_all_lattices():
    return get_all_lattices_from('Resources/lattices/')


def load_lattice(lattice_name):
    return Lattice.load_from_pickle('Resources/lattices/' + lattice_name + '.pkl')


def compute_all_quadratic_form_per_shell_parallel(lattice):
    print(lattice.name)
    lattice.compute_eigenvalues_per_length()
    lattice.save_as_pickle('Resources/lattices')
    print(f'done with ' + lattice.name)


def run_parallel(method, inputs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(method, inputs)

