from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from qndiag import ajd_pham


class Solver(BaseSolver):
    name = 'Pham'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/pierreablin/qndiag@master'
    ]

    def set_objective(self, C, ortho):
        self.C = C
        self.ortho = ortho

    def skip(self, C, ortho):
        if ortho:
            return True, "Pham does not support orthogonal constraint."
        return False, None

    def run(self, n_iter):
        self.B, _ = ajd_pham(self.C, max_iter=n_iter)

    def get_result(self):
        return self.B
