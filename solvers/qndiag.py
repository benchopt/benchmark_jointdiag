from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from qndiag import qndiag


class Solver(BaseSolver):
    name = 'qndiag'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/pierreablin/qndiag@master'
    ]

    def set_objective(self, C):
        self.C = C

    def run(self, n_iter):
        self.B, _ = qndiag(self.C, max_iter=n_iter)

    def get_result(self):
        return self.B
