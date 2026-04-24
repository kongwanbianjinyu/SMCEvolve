# First Autocorrelation Inequality

AlphaEvolve Appendix B.1. Construct a non-negative step function
`f: R → R` to improve the upper bound on a constant C1 related to
the autoconvolution of `f`.

The program must expose:

```python
def run():
    # returns (f_values, c1_achieved, loss, n_points)
    ...
```

Reward = `BENCHMARK / c1_achieved`. Maximize.
