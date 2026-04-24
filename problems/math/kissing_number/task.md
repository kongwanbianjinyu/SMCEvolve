# 11-Dimensional Kissing Number

AlphaEvolve Appendix B.11. Construct an integer point cloud in
R^11 such that, after rounding to integers,
`min pairwise squared distance ≥ max squared norm`. The number of
points is a lower bound on the 11-D kissing number.

Define:

```python
def kissing_number11() -> np.ndarray:
    """Return an (N, 11) integer array with N >= 2."""
```

Reward = `N / 593` (AlphaEvolve benchmark, 2024).
Invalid output (contains origin, fails the norm/distance
inequality) → 0.0. Maximize.
