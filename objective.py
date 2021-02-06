import numpy as np
from benchopt import BaseObjective


def transform_set(M, D):
    n_matrices, n_features = D.shape[:2]
    op = np.zeros((n_matrices, n_features, n_features))
    for i, d in enumerate(D):
        op[i] = M.dot(d.dot(M.T))
    return op


class Objective(BaseObjective):
    name = "Joint Diagonalization"

    def __init__(self):
        pass

    def set_data(self, C):
        self.C = C

    def compute(self, B):
        D = transform_set(B, self.C)
        n_matrices, n_features = self.C.shape[:2]
        diagonals = np.diagonal(D, axis1=1, axis2=2)
        logdet = -np.linalg.slogdet(B)[1]
        return logdet + 0.5 * np.sum(np.log(diagonals)) / n_matrices

    def to_dict(self):
        return dict(C=self.C)
