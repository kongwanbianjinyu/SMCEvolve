"""Constructor-based circle packing for n=26 circles in the unit square."""

import numpy as np

N = 26


def construct_packing():
    """Place 26 circles: one center + 8-ring + 16-outer-ring."""
    centers = np.zeros((N, 2))

    centers[0] = [0.500000, 0.500000]

    for i in range(8):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.500000 + 0.300000 * np.cos(angle),
                          0.500000 + 0.300000 * np.sin(angle)]

    for i in range(16):
        angle = 2 * np.pi * i / 16
        centers[i + 9] = [0.500000 + 0.700000 * np.cos(angle),
                          0.500000 + 0.700000 * np.sin(angle)]

    centers = np.clip(centers, 0.010000, 0.990000)
    radii = compute_max_radii(centers)
    return centers, radii


def compute_max_radii(centers):
    """Greedy radii: clip by border, then iteratively shrink overlapping pairs."""
    n = centers.shape[0]
    radii = np.array([min(x, y, 1.0 - x, 1.0 - y) for x, y in centers], dtype=float)
    radii = np.maximum(radii, 0.0)

    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.sqrt(np.sum((centers[i] - centers[j]) ** 2)))
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break
    return np.maximum(radii, 0.0)


def run_packing():
    centers, radii = construct_packing()
    return centers, radii, float(np.sum(radii))
