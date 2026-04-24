# Hexagon Packing (n = 11)

AlphaEvolve Appendix B.7. Place 11 disjoint unit regular hexagons
inside a larger regular hexagon. Minimize the outer hexagon's
side length (equivalently, maximize `1 / outer_side_length`).

Define:

```python
def hexagon_packing_11():
    """Return (inner_hex_data, outer_hex_data, outer_side_length).

    inner_hex_data : (11, 3) array of (x, y, angle_degrees) for each unit hex.
    outer_hex_data : (3,)    array (cx, cy, angle_degrees) for the outer hex.
    outer_side_length : float > 0.
    """
```

Reward = `(1 / outer_side_length) / (1 / 3.930092)`. Invalid
configurations (overlap, inner hexagons outside the outer one) → 0.0.
Maximize.
