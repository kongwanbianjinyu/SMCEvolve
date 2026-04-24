# Heilbronn Triangle Problem (n = 11)

AlphaEvolve Appendix B.9. Place 11 points inside the unit
equilateral triangle with vertices `(0, 0)`, `(1, 0)`,
`(0.5, sqrt(3)/2)` so that the smallest triangle formed by any
3 of the points has maximum area.

Define:

```python
def heilbronn_triangle11() -> np.ndarray:
    """Return an (11, 2) array of (x, y) coordinates inside the triangle."""
```

Reward = `min_triangle_area_normalized / 0.036529889880030156`.
Invalid output (out-of-triangle points, wrong shape) → 0.0.
Maximize.
