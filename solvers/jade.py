from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import pyriemann
    if pyriemann.__version__ < "0.13":
        from pyriemann.utils.ajd import rjd as jade
        transpose_out = True
    else:
        from pyriemann.geometry.ajd import jade
        transpose_out = False


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
        self.B, _ = jade(self.C, n_iter_max=n_iter)
        if transpose_out:
            self.B = self.B.T

    def get_result(self):
        return dict(B=self.B)
