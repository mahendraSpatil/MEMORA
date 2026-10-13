"""Retention value, normalisation, and the value-density eviction key.

The proposal's original score added size to the value terms:

    F(m) = w1*A + w2*S + w3(1-I) + w4(1-R) + w5(1-C)

That treats "this file is big" as evidence the memory is worthless.  It is
not: size is the *benefit* of deleting, the other terms are the *cost*.
Mixing them into one weighted sum leaves the weights uninterpretable.

Stated properly, the task is to free at least B bytes while losing as little
value as possible -- a minimum-cost covering knapsack, which is NP-hard.  So
a greedy is the right tool, and the right greedy key is value per byte:

    V(m) = (1 - w_C) * [ utility terms ]  +  w_C * C(m)
    rho(m) = V(m) / size(m)          -> evict in increasing rho

Size needs no weight of its own, which drops five tunable numbers to four,
and ``w_C = 0`` gives the value-greedy ablation for free: the identical
policy with the graph switched off.

Everything is rank-normalised rather than min-max normalised.  Min-max lets
one 2 GB outlier crush every other size to ~0; percentile rank costs
O(V log V) and is robust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import MemoryStore

# Relative split of the non-structural terms.  Rescaled by (1 - w_C) so that
# sweeping the context weight never changes the total scale of the value.
_BASE = np.array([0.35, 0.20, 0.15])
_BASE = _BASE / _BASE.sum()  # importance, usage, freshness


def rank_normalize(a: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] with ties averaged."""
    n = len(a)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks / (n - 1)


@dataclass
class StaticFeatures:
    """The part of the value that never changes as the graph shrinks."""

    importance: np.ndarray
    usage: np.ndarray
    freshness: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        return (
            _BASE[0] * self.importance
            + _BASE[1] * self.usage
            + _BASE[2] * self.freshness
        )


def static_features(store: MemoryStore) -> StaticFeatures:
    """Rank-normalise the metadata a policy is allowed to see.

    Note this reads ``observed_importance``, never ``true_importance``.
    """
    importance = rank_normalize(store.observed_importance)
    recency = 1.0 - rank_normalize(store.recency_days)
    frequency = rank_normalize(np.log1p(store.access_count))
    usage = 0.5 * recency + 0.5 * frequency
    freshness = 1.0 - rank_normalize(store.age_days)
    return StaticFeatures(importance, usage, freshness)


def structural_score(
    damage: np.ndarray,
    weighted_degree: np.ndarray,
    damage_scale: float,
    max_weighted_degree: float,
    *,
    cut_weight: float = 0.7,
    transform: str = "sqrt",
) -> np.ndarray:
    """Turn raw pair damage into a context score C(v) in [0, 1].

    ``damage_scale`` is the largest damage present in the *initial* graph.
    Normalising against total pairs instead looks more principled but crushes
    almost every node to ~0: in a 1,000-node graph, severing a five-node
    cluster destroys 1% of the pairs, so the term stops influencing anything.
    Calibrating against the worst case actually present keeps the full [0, 1]
    range available, and fixing the scale at t=0 keeps it stable while the
    graph shrinks.

    ``cut_weight`` splits C between disconnection damage and the local
    context a node carries in its own edges.  Setting it to 0 gives the
    degree-only baseline, which is the cheap graph signal CAFA has to beat to
    justify running Tarjan at all.

    The transform is stateless on purpose: rank-normalising inside the
    eviction loop would cost O(V log V) per deletion.  ``sqrt`` spreads the
    lower end, where almost every node sits, without needing global state.
    """
    d = damage / max(damage_scale, 1.0)
    if transform == "sqrt":
        d = np.sqrt(np.clip(d, 0.0, 1.0))
    elif transform == "linear":
        d = np.clip(d, 0.0, 1.0)
    elif transform == "log":
        d = np.log1p(np.clip(d, 0.0, 1.0) * 9.0) / np.log(10.0)
    else:
        raise ValueError(f"unknown transform {transform!r}")
    if max_weighted_degree > 0:
        loc = np.minimum(1.0, weighted_degree / max_weighted_degree)
    else:
        loc = np.zeros_like(d)
    return cut_weight * d + (1.0 - cut_weight) * loc


def retention_value(
    static_part: np.ndarray, context: np.ndarray, w_context: float
) -> np.ndarray:
    return (1.0 - w_context) * static_part + w_context * context


def value_density(value: np.ndarray, size_mb: np.ndarray) -> np.ndarray:
    return value / np.maximum(size_mb, 1e-9)
