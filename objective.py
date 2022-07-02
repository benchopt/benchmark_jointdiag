import numpy as np
from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Joint Diagonalization"

    parameters = {
        'ortho': [False, True]
    }

    def __init__(self, ortho=False):
        self.ortho = ortho

    def set_data(self, C):
        self.C = C

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
        obj = np.sum(np.log(diagonals))
        for Di in D:
            obj -= np.linalg.slogdet(Di)[1]
        obj /= 2 * n_matrices
        obj = np.exp(obj)  # take exp for stopping criterion to work better
        return obj

    def to_dict(self):
        return dict(C=self.C, ortho=self.ortho)
