from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyriemann.utils.ajd import rjd


class Solver(BaseSolver):
    name = "Jade"

    install_cmd = 'conda'
    requirements = ['pyriemann']

    def set_objective(self, C, ortho):
        self.C = C
        self.ortho = ortho

    def skip(self, C, ortho):
        if not ortho:
            return True, "Jade supports only orthogonal constraint."
        return False, None

    def run(self, n_iter):
        self.B, _ = rjd(self.C, n_iter_max=n_iter)

    def get_result(self):
        return dict(B=self.B)
