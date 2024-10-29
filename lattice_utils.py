import numpy as np
from numpy import sqrt


def build_basis_tsm(n):
    basis = []
    for i in range(0, n):
        for j in range(i + 1, n):
            h = np.zeros((n, n))
            h[i, j] = 0.5
            h[j, i] = 0.5
            basis.append(h)

    nobasis = []
    for i in range(0, n - 1):
        h = np.zeros((n, n))
        h[i, i] = 0.5
        h[n - 1, n - 1] = -0.5
        nobasis.append(h)

    gsbasis = []
    for i in range(0, n - 1):
        bstar = nobasis[i]
        for j in range(0, i):
            bstar = bstar - (gsbasis[j] * nobasis[i]).trace() / (gsbasis[j] * gsbasis[j]).trace() * gsbasis[j]
        gsbasis.append(bstar)

    for b in gsbasis:
        basis.append(1 / sqrt(2 * (b * b).trace()) * b)

    return basis