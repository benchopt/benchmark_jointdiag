import numpy as np
from benchopt import BaseObjective


def amari_distance(W, A):
    """
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.

    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix

    A : ndarray, shape (n_features, n_features)
        Input matrix

    Returns
    -------
    d : float
        The Amari distance
    """
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)
    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


class Objective(BaseObjective):
    name = "Joint Diagonalization"

    parameters = {
        'ortho': [False, True]
    }

    def __init__(self, ortho=False):
        self.ortho = ortho

    def set_data(self, C, A):
        self.C = C
        self.A = A

    def get_one_solution(self):
        return np.eye((self.C.shape[1]))

    def compute(self, B):
        if self.ortho:
            is_ortho = np.linalg.norm(
                B @ B.T - np.eye(len(B)), ord=np.inf) < 1e-6
            if not is_ortho:
                return np.inf  # constraint is violated
        D = B @ self.C @ B.T  # diagonalization with B
        n_matrices, _, _ = D.shape
        diagonals = np.diagonal(D, axis1=1, axis2=2)
        nll = np.sum(np.log(diagonals))
        for Di in D:
            nll -= np.linalg.slogdet(Di)[1]
        nll /= 2 * n_matrices
        obj = np.exp(nll)  # take exp for stopping criterion to work better
        return {
            'value': obj,
            'NLL': nll,  # negative log-likelihood
            'Amari distance': amari_distance(B, self.A)
        }

    def to_dict(self):
        return dict(C=self.C, ortho=self.ortho)
