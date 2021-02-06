from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from qndiag import ajd_pham


class Solver(BaseSolver):
    name = 'Pham'

    install_cmd = 'conda'
    requirements = [
        'pip:https://api.github.com/repos/pierreablin/qndiag/master']

    def set_objective(self, C):
        self.C = C

    def run(self, n_iter):
        self.B, _ = ajd_pham(self.C, max_iter=n_iter)

    def get_result(self):
        return self.B
