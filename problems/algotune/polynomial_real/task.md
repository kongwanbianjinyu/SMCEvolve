# AlgoTune: polynomial_real

PolynomialReal Task:

Given a polynomial of degree n with all real roots, the task is to approximate the roots of the polynomial.
The goal is to compute the approximated roots so that each is within 0.001 of the true value—i.e., correct to at least three decimal places (any extra precision beyond three decimal places is not considered).
A given solution's distance is said to be the average absolute difference between the approximated roots and the true roots.

Input: A list of polynomial coefficients in descending order.

Example input:
[1.0, -3.54, 3.25, -1.17, 0.23],

(The polynomial is x^4 - 3.54x^3 + 3.25x^2 - 1.17x + 0.23)

Output: A list of approximated roots in descending order.

Example output:
[1.544, 1.022, -0.478, -1.273]

Category: numerical_methods

---

Define a class `<TaskName>` with a `solve(self, problem)` method and expose a module-level `run_solver(problem)` entry point.

Reward = AlgoTune speedup vs. the reference `solve` implementation (primary fitness = `speedup_score`). Invalid solutions score 0.0. See `openevolve_evaluator.py` for the exact timing harness and validity predicate.
