"""Reward = sum of radii for n=26 circles packed in the unit square.

Returns 0.0 when the candidate program crashes, omits run_packing(),
returns malformed shapes, places a circle outside the unit square, or
produces overlapping circles.
"""

import math

N = 26
EPS = 1e-6


def evaluate(program: str) -> float:
    namespace: dict = {"__name__": "__main__"}
    try:
        exec(program, namespace)
    except Exception:
        return 0.0

    run_packing = namespace.get("run_packing")
    if not callable(run_packing):
        return 0.0

    try:
        result = run_packing()
        centers, radii, _reported = result
        centers = [(float(x), float(y)) for x, y in centers]
        radii = [float(r) for r in radii]
    except Exception:
        return 0.0

    if len(centers) != N or len(radii) != N:
        return 0.0

    for r in radii:
        if not math.isfinite(r) or r < 0:
            return 0.0
    for x, y in centers:
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0

    for (x, y), r in zip(centers, radii):
        if x - r < -EPS or x + r > 1.0 + EPS:
            return 0.0
        if y - r < -EPS or y + r > 1.0 + EPS:
            return 0.0

    for i in range(N):
        xi, yi = centers[i]
        for j in range(i + 1, N):
            xj, yj = centers[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < radii[i] + radii[j] - EPS:
                return 0.0

    return sum(radii)
