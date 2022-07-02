from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


loss = import_ctx.import_from('common', 'loss')


class Solver(BaseSolver):
    name = "qndiag"
    stopping_strategy = "callback"

    references = [
        "P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham's algorithm"
        "for joint diagonalization. Proc. ESANN 2019."
        "https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf"
        "https://hal.archives-ouvertes.fr/hal-01936887v1"
        "https://arxiv.org/abs/1811.11433"
    ]

    def set_objective(self, C):
        self.C = C

    def run(self, callback):
        C = self.C
        _, n_features, _ = C.shape

        def _linesearch(D, B, direction, current_loss, n_ls_tries):
            _, p, _ = D.shape
            step = 1.0
            for _ in range(n_ls_tries):
                M = np.eye(p) + step * direction
                new_B = np.dot(M, B)
                new_loss = loss(C, new_B)
                if new_loss < current_loss:
                    success = True
                    break
                step /= 2.0
            else:
                success = False
            return success, new_B, new_loss, step * direction

        # init B
        C_mean = np.mean(C, axis=0)
        d, p = np.linalg.eigh(C_mean)
        B = p.T / np.sqrt(d[:, None])

        lambda_min = 1e-4
        n_ls_tries = 10

        obj = loss(C, B)

        while callback(B):
            # Gradient
            D = B @ C @ B.T  # diagonalization with B
            diags = np.diagonal(D, axis1=1, axis2=2)
            G = np.average(D / diags[:, :, None], axis=0) - np.eye(n_features)

            # Hessian coefficients
            h = np.average(diags[:, None, :] / diags[:, :, None], axis=0)

            det = h * h.T - 1.0
            det[det < lambda_min] = lambda_min  # Regularize
            direction = -(G * h.T - G.T) / det
            _, B, obj, _ = _linesearch(D, B, direction, obj, n_ls_tries)

        self.B = B

    def get_result(self):
        return self.B
