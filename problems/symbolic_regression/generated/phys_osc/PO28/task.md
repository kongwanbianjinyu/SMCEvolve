# Symbolic Regression: phys_osc / PO28

Target output: **dv_dt**.
Input features (columns of `x`, in order): **x, t, v**.

Write a Python function

```python
def func(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    '''x: (n_samples, 3)   params: (10,)   returns: (n_samples,)'''
```

and export `run_search()` returning `func`. Parameters are
BFGS-optimized externally; reward is `-log10(train_mse + 1e-9)`
(higher is better). Any runtime error / NaN / shape mismatch → 0.0.
