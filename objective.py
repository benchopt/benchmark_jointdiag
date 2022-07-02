import numpy as np
from benchopt import BaseObjective
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    pass


loss = import_ctx.import_from('common', 'loss')


class Objective(BaseObjective):
    name = "Joint Diagonalization"

    def __init__(self):
        pass

    def set_data(self, C):
        self.C = C

    def get_one_solution(self):
        return np.eye((self.C.shape[1]))

    def compute(self, B):
        return np.exp(loss(self.C, B))

    def to_dict(self):
        return dict(C=self.C)
