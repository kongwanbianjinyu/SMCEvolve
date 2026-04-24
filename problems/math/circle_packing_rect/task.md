# Circle Packing in a Perimeter-4 Rectangle (n = 21)

AlphaEvolve Appendix B.13. Place 21 disjoint circles inside an
axis-aligned rectangle of perimeter 4 (width + height ≤ 2) to
maximize the sum of radii.

Define:

```python
def circle_packing21() -> np.ndarray:
    """Return a (21, 3) array whose row i is (x, y, r)."""
```

The evaluator (`openevolve_evaluator.py`) enforces:
- Shape `(21, 3)`, finite floats, non-negative radii.
- No pair of circles overlaps.
- Minimum enclosing axis-aligned rectangle has
  `width + height ≤ 2 + 1e-6`.

Reward = `sum(radii) / 2.3658321334167627` (AlphaEvolve benchmark).
Invalid output → 0.0. Maximize.
