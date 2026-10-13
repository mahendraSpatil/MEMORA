"""Evaluation metrics.

The proposal's original context measure -- ``1 - fragmentation_after /
max_possible_fragmentation`` -- has two problems.  Maximum possible
fragmentation is not well defined, and a component count is not proportional
to damage: splitting off one leaf and halving the graph both add exactly one
component.

Everything here is therefore computed **relative to the survivors**.  Take
the set S of memories a policy kept; the denominator is how many pairs of S
were connected in the original graph, and the numerator is how many still
are.  Deleting a memory that lay on no path between survivors then costs
nothing, which is the correct behaviour and is not what a raw component count
does.

Three validity threats the metrics here are built to close:

1. *Importance is both an input and a score.*  Policies read
   ``observed_importance``; ``importance_retained`` scores
   ``true_importance``.  Sweep the noise between them.
2. *The objective and the metric are the same quantity.*  CAFA minimises pair
   damage and ``pairwise_retention`` measures pair damage, so
   ``context_query_success`` is reported alongside it: a held-out sample of
   related pairs drawn **before** any deletion, which CAFA never optimised.
3. *Unequal work.*  ``overshoot`` and ``n_deleted`` are reported on every row
   so an unfair comparison is visible rather than hidden.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .algorithms.base import EvictionResult
from .graph import MemoryGraph
from .model import MemoryStore
from .tarjan import connected_components


def sample_context_queries(
    graph: MemoryGraph, *, n_queries: int = 500, max_depth: int = 3, seed: int = 0
) -> list[tuple[int, int]]:
    """Sample related pairs from the intact graph.

    Sampled once per dataset and shared by every policy, so all of them are
    graded on exactly the same questions.  Pairs are drawn at distance 2-3:
    adjacent pairs are too easy, and unrelated pairs measure nothing.
    """
    rng = np.random.default_rng(seed)
    n = graph.n
    pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(pairs) < n_queries and attempts < n_queries * 25:
        attempts += 1
        u = int(rng.integers(n))
        seen = {u: 0}
        frontier = [u]
        far: list[int] = []
        for d in range(1, max_depth + 1):
            nxt = []
            for x in frontier:
                for y in graph.adj[x]:
                    if y not in seen:
                        seen[y] = d
                        nxt.append(y)
                        if d >= 2:
                            far.append(y)
            frontier = nxt
            if not frontier:
                break
        if far:
            pairs.append((u, int(far[int(rng.integers(len(far)))])))
    return pairs


def evaluate(
    store: MemoryStore,
    original: MemoryGraph,
    result: EvictionResult,
    queries: list[tuple[int, int]] | None = None,
) -> dict:
    """Score one policy run.  O(V + E)."""
    g = result.graph
    alive = g.alive
    survivors = np.flatnonzero(alive)

    orig_comp = original.original_components()
    orig_groups = Counter(orig_comp[survivors].tolist())
    _, new_sizes = connected_components(g)

    pairs_before = sum(c * (c - 1) // 2 for c in orig_groups.values())
    pairs_after = sum(c * (c - 1) // 2 for c in new_sizes.values())
    pairwise_retention = (pairs_after / pairs_before) if pairs_before else 1.0

    lcc_before = max(orig_groups.values()) if orig_groups else 0
    lcc_after = max(new_sizes.values()) if new_sizes else 0
    lcc_ratio = (lcc_after / lcc_before) if lcc_before else 1.0

    # Neighbourhood retention uses the ORIGINAL adjacency for both sides, so
    # repair edges added by CAFA-R cannot inflate it.
    total, counted = 0.0, 0
    for v in survivors:
        d = original.adj[v]
        if not d:
            continue
        total += sum(1 for w in d if alive[w]) / len(d)
        counted += 1
    neighbourhood_retention = (total / counted) if counted else 1.0

    importance_retained = float(
        store.true_importance[survivors].sum() / store.true_importance.sum()
    )

    metrics = {
        "policy": result.policy,
        "n_deleted": result.n_deleted,
        "bytes_freed_mb": round(result.bytes_freed, 3),
        "required_mb": round(result.required_mb, 3),
        "overshoot": round(result.overshoot, 4),
        "runtime_s": round(result.runtime_s, 6),
        "pairwise_retention": round(pairwise_retention, 5),
        "neighbourhood_retention": round(neighbourhood_retention, 5),
        "lcc_ratio": round(lcc_ratio, 5),
        "components_after": len(new_sizes),
        "components_before": len(orig_groups),
        "importance_retained": round(importance_retained, 5),
        "survivors": int(len(survivors)),
    }

    bridges = np.array([m.is_bridge for m in store.memories], dtype=bool)
    if bridges.any():
        metrics["bridge_retention"] = round(
            float(alive[bridges].sum() / bridges.sum()), 5
        )

    if queries:
        new_comp, _ = connected_components(g)
        both_alive = 0
        still_linked = 0
        for u, w in queries:
            if alive[u] and alive[w]:
                both_alive += 1
                if new_comp[u] == new_comp[w] and new_comp[u] != -1:
                    still_linked += 1
        metrics["context_query_success"] = round(
            still_linked / both_alive, 5) if both_alive else 1.0
        metrics["context_query_survival"] = round(still_linked / len(queries), 5)
        metrics["query_coverage"] = round(both_alive / len(queries), 5)

    metrics.update({f"stat_{k}": v for k, v in result.stats.items()})
    return metrics
