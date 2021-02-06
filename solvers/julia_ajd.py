from pathlib import Path
from benchopt import safe_import_context

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = str(Path(__file__).with_suffix('.jl'))


class Solver(JuliaSolver):

    # Config of the solver
    name = 'Julia-AJD'
    stop_strategy = 'iteration'

    def skip(self, C):
        return False, "Not Implemented"

    def set_objective(self, C):
        self.C = C

        jl = get_jl_interpreter()
        self.solve_ajd = jl.include(JULIA_SOLVER_FILE)

    def run(self, n_iter):
        self.B = self.solve_ajd(self.C, n_iter)

    def get_result(self):
        return self.B
