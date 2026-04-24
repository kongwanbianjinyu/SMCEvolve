# Heilbronn Problem for Convex Regions (n = 13)

AlphaEvolve Appendix B.10. Place 13 points inside a convex region
of unit area so that the smallest triangle formed by any 3 points
has maximum area. The evaluator supplies the convex region.

Define:

```python
def heilbronn_convex13() -> np.ndarray:
    """Return a (13, 2) array of point coordinates inside the region."""
```

Reward = `min_area_normalized / BENCHMARK`. Maximize.
