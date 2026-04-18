"""Adaptive temperature schedule via ESS bisection (Sec 2.3 of the paper).

ESS(λ) is strictly decreasing in λ, so a unique solution to ESS(λ) = κN
exists in (λ_prev, 1] whenever ESS(1) < κN.
"""

from __future__ import annotations

import math
from typing import Sequence


def ess(rewards: Sequence[float], delta_beta: float) -> float:
    if delta_beta == 0.0:
        return float(len(rewards))
    log_w = [delta_beta * r for r in rewards]
    m = max(log_w)
    w = [math.exp(lw - m) for lw in log_w]
    s1 = sum(w)
    s2 = sum(wi * wi for wi in w)
    return (s1 * s1) / s2


def find_next_lambda(
    rewards: Sequence[float],
    lam_prev: float,
    beta_target: float,
    kappa: float,
    lam_max: float = 1.0,
    max_delta: float | None = None,
    tol: float = 1e-4,
) -> float:
    n = len(rewards)
    target = kappa * n

    if max_delta is not None:
        lam_max = min(lam_max, lam_prev + max_delta)

    def ess_at(lam: float) -> float:
        return ess(rewards, (lam - lam_prev) * beta_target)

    if ess_at(lam_max) >= target:
        return lam_max

    lo, hi = lam_prev, lam_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo
