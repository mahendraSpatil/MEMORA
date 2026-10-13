"""Shared plumbing for every eviction policy.

The one rule that makes the comparison meaningful: **every policy must free
at least the same number of bytes from the same input.**  Otherwise CAFA can
look good purely by deleting five large items where LRU deleted forty small
ones, and the results table measures nothing.  Every policy therefore runs
through :func:`evict_in_order` or reports ``bytes_freed`` and ``overshoot``
so the fairness of the run is visible in the output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..graph import MemoryGraph
from ..model import MemoryStore


@dataclass
class EvictionResult:
    policy: str
    deleted: list[int]
    bytes_freed: float
    required_mb: float
    runtime_s: float
    graph: MemoryGraph
    stats: dict = field(default_factory=dict)

    @property
    def overshoot(self) -> float:
        return self.bytes_freed / self.required_mb if self.required_mb else 1.0

    @property
    def n_deleted(self) -> int:
        return len(self.deleted)


def evict_in_order(
    name: str,
    store: MemoryStore,
    graph: MemoryGraph,
    required_mb: float,
    order: np.ndarray | list[int],
    started: float | None = None,
) -> EvictionResult:
    """Delete along a precomputed priority order until the budget is met."""
    started = started if started is not None else time.perf_counter()
    freed = 0.0
    deleted: list[int] = []
    sizes = store.size_mb
    for v in order:
        if freed >= required_mb:
            break
        v = int(v)
        graph.delete(v)
        deleted.append(v)
        freed += sizes[v]
    return EvictionResult(
        policy=name,
        deleted=deleted,
        bytes_freed=freed,
        required_mb=required_mb,
        runtime_s=time.perf_counter() - started,
        graph=graph,
    )


def repair_overshoot(
    store: MemoryStore,
    graph: MemoryGraph,
    deleted: list[int],
    value_lost: dict[int, float],
    freed: float,
    required_mb: float,
) -> tuple[list[int], float, int]:
    """Undo deletions the budget did not need.

    Greedy overshoots.  Budget 10 MB; X is 6 MB worth 0.30, Y is 10 MB worth
    0.60, Z is 4 MB worth 0.28.  Greedy takes X (density 0.050) then Y
    (0.060): 16 MB freed, 0.90 lost.  X and Z free exactly 10 MB for 0.58.

    Walking the deleted list in decreasing value and restoring anything that
    fits is O(K log K) and can never make the result worse.
    """
    restored = 0
    for v in sorted(deleted, key=lambda i: -value_lost.get(i, 0.0)):
        if freed - store.size_mb[v] >= required_mb:
            graph.restore(v)
            freed -= float(store.size_mb[v])
            restored += 1
    if restored:
        deleted = [v for v in deleted if graph.alive[v] == False]  # noqa: E712
    return deleted, freed, restored
