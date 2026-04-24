# AlgoTune: lu_factorization

LUFactorization Task:

Given a square matrix A, the task is to compute its LU factorization.
The LU factorization decomposes A as:

    A = P · L · U

where P is a permutation matrix, L is a lower triangular matrix with ones on the diagonal, and U is an upper triangular matrix.

Input: A dictionary with key:
  - "matrix": An array representing the square matrix A. (The dimension n is inferred from the matrix.)

Example input:
{
    "matrix": [
        [2.0, 3.0],
        [5.0, 4.0]
    ]
}

Output: A dictionary with a key "LU" that maps to another dictionary containing three matrices:
- "P" is the permutation matrix that reorders the rows of A.
- "L" is the lower triangular matrix with ones on its diagonal.
- "U" is the upper triangular matrix.
These matrices satisfy the equation A = P L U.

Example output:
{
    "LU": {
        "P": [[0.0, 1.0], [1.0, 0.0]],
        "L": [[1.0, 0.0], [0.4, 1.0]],
        "U": [[5.0, 4.0], [0.0, 1.4]]
    }
}

Category: matrix_operations

---

Define a class `<TaskName>` with a `solve(self, problem)` method and expose a module-level `run_solver(problem)` entry point.

Reward = AlgoTune speedup vs. the reference `solve` implementation (primary fitness = `speedup_score`). Invalid solutions score 0.0. See `openevolve_evaluator.py` for the exact timing harness and validity predicate.
