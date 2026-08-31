from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pyriemann
    if pyriemann.__version__ < "0.13":
        from pyriemann.utils.ajd import ajd_pham
    else:
        from pyriemann.geometry.ajd import ajd_pham


class Solver(BaseSolver):
    name = "Pham"

    install_cmd = 'conda'
    requirements = ['pyriemann']

    def set_objective(self, C, ortho):
        self.C = C
        self.ortho = ortho

    def skip(self, C, ortho):
        if ortho:
            return True, "Pham does not support orthogonal constraint."
        return False, None

    def run(self, n_iter):
        self.B, _ = ajd_pham(self.C, n_iter_max=n_iter)

    def get_result(self):
        return dict(B=self.B)
