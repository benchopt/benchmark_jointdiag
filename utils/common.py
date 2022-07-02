import numpy as np


def loss(C, B):
    D = B @ C @ B.T  # diagonalization with B
    n_matrices, _, _ = C.shape
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    logdet = -np.linalg.slogdet(B)[1]
    obj = logdet + 0.5 * np.sum(np.log(diagonals)) / n_matrices
    return obj
