from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from pyriemann.utils.ajd import uwedge


class Solver(BaseSolver):
    name = "U-WEDGE"

    install_cmd = 'conda'
    requirements = ['pyriemann']

    def set_objective(self, C, ortho):
        self.C = C
        self.ortho = ortho

    def skip(self, C, ortho):
        if ortho:
            return True, "U-WEDGE does not support orthogonal constraint."
        return False, None

    def run(self, n_iter):
        self.B, _ = uwedge(self.C, n_iter_max=n_iter)

    def get_result(self):
        return dict(B=self.B)
