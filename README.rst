Benchmark repository for Approximate Joint Diagonalization
==========================================================

|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark considers the approximate joint diagonalization (AJD)
of positive matrices. Given $n$ square symmetric positive matrices $C^i$,
it consists of solving the following problem:

$$
\\min_B \\frac{1}{2n} \\sum_{i=1}^n \\log | \\textrm{diag} (B C^i B^{\\top}) | - \\log | B C^i B^{\\top} |
$$

where | | stands for the matrix determinant and $\\textrm{diag}$ stands
for the operator that keeps only the diagonal elements of a matrix. Optionally,
the matrix $B$ can be enforced to be orthogonal.

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_jointdiag
   $ benchopt run ./benchmark_jointdiag


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

Troubleshooting
---------------

If you run into some errors when running the examples present in this Readme,
try installing the development version of `benchopt`:

.. code-block::

  $ pip install -U git+https://github.com/benchopt/benchopt


.. |Build Status| image:: https://github.com/benchopt/benchmark_jointdiag/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_jointdiag/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
