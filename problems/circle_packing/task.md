# Circle Packing (n=26)

Define `run_packing()` returning `(centers, radii, sum_of_radii)`:

- `centers`: shape (26, 2) array of (x, y) coordinates in [0, 1]²
- `radii`:   shape (26,) array of non-negative radii
- `sum_of_radii`: scalar (recomputed by the evaluator anyway)

Constraints:
- every circle must lie inside the unit square (`x ± r ∈ [0,1]`, same for y)
- no two circles may overlap

Reward = ∑ radii. Maximize.

