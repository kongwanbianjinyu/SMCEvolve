# Minimizing Max/Min Pairwise Distance (n = 16, d = 2)

AlphaEvolve Appendix B.8. Place 16 points in R² so that the ratio
between the maximum and minimum pairwise distance is as small as
possible (equivalently, maximize the squared inverse ratio).

Define:

```python
def min_max_dist_dim2_16() -> np.ndarray:
    """Return a (16, 2) array of point coordinates."""
```

Reward = `inv_ratio_squared / BENCHMARK`. Maximize.
