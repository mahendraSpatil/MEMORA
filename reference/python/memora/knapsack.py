"""Exact reference for small instances.

Freeing at least B bytes at minimum value lost is a minimum-cost covering
knapsack: NP-hard, which is what justifies the greedy.  But "greedy is
reasonable because the problem is hard" is a much weaker claim than "greedy
lands within 4% of optimal on 100-node instances", and the second one costs
an O(V*B) dynamic program to obtain.

Run this on the 100- and 200-memory datasets only and report the gap.
"""

from __future__ import annotations

import math

import numpy as np

INF = float("inf")


def optimal_value_lost(
    sizes_mb: np.ndarray, values: np.ndarray, required_mb: float, *, scale: float = 1.0
) -> float:
    """Minimum total value lost by any subset freeing at least ``required_mb``.

    Sizes are discretised at ``scale`` units per MB; raising it trades runtime
    for precision.
    """
    units = np.maximum(1, np.rint(sizes_mb * scale).astype(np.int64))
    budget = int(math.ceil(required_mb * scale))
    if budget <= 0:
        return 0.0
    if int(units.sum()) < budget:
        return float(values.sum())

    dp = [INF] * (budget + 1)
    dp[0] = 0.0
    for u, val in zip(units.tolist(), values.tolist()):
        for j in range(budget, -1, -1):
            cur = dp[j]
            if cur == INF:
                continue
            k = budget if j + u > budget else j + u
            cand = cur + val
            if cand < dp[k]:
                dp[k] = cand
    return dp[budget]


def brute_force_value_lost(
    sizes_mb: np.ndarray, values: np.ndarray, required_mb: float
) -> float:
    """Exhaustive reference for the test suite.  Use with <= 16 items."""
    n = len(sizes_mb)
    if n > 20:
        raise ValueError("brute force is for tiny instances only")
    best = INF
    for mask in range(1 << n):
        freed = 0.0
        lost = 0.0
        m = mask
        i = 0
        while m:
            if m & 1:
                freed += sizes_mb[i]
                lost += values[i]
            m >>= 1
            i += 1
        if freed >= required_mb and lost < best:
            best = lost
    return best
