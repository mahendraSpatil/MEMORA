"""The memory graph and its construction.

Two things in here matter for the report.

**Construction is the real bottleneck, not Tarjan.**  Comparing every pair of
memories against a similarity threshold is O(V^2): 5x10^7 comparisons at
V=10,000, which costs more than every eviction run in the benchmark put
together.  :func:`build_graph` blocks by category and samples a fixed number
of cross-category candidates instead, so the cost is
O(sum_c |cat_c|^2 + V*cross_samples).

**Density is a parameter, not an accident.**  The builder keeps a mutual
nearest-neighbour backbone and then fills a global edge budget, so
``target_degree`` is hit almost exactly.  Average degree is the variable that
decides whether structural awareness pays off at all, so it has to be dialled
directly rather than chased through a similarity threshold.

Deletion is a flag flip.  Adjacency dictionaries are never mutated on delete,
which makes both deletion and restoration O(1) and keeps the repair pass free.
"""

from __future__ import annotations

import numpy as np

from .model import MemoryStore


class MemoryGraph:
    """Undirected weighted graph over contiguous integer node ids."""

    __slots__ = ("n", "adj", "alive", "_orig_comp")

    def __init__(self, n: int):
        self.n = n
        self.adj: list[dict[int, float]] = [dict() for _ in range(n)]
        self.alive = np.ones(n, dtype=bool)
        self._orig_comp: np.ndarray | None = None

    # ---- construction -------------------------------------------------
    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        if u == v:
            return
        self.adj[u][v] = w
        self.adj[v][u] = w

    # ---- queries ------------------------------------------------------
    def neighbours(self, v: int) -> list[int]:
        alive = self.alive
        return [w for w in self.adj[v] if alive[w]]

    def degree(self, v: int) -> int:
        alive = self.alive
        return sum(1 for w in self.adj[v] if alive[w])

    def weighted_degree(self, v: int) -> float:
        alive = self.alive
        return sum(w for u, w in self.adj[v].items() if alive[u])

    def live_nodes(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    @property
    def n_live(self) -> int:
        return int(self.alive.sum())

    def edge_count(self) -> int:
        return sum(
            1 for v in range(self.n) if self.alive[v]
            for w in self.adj[v] if self.alive[w] and w > v
        )

    # ---- mutation -----------------------------------------------------
    def delete(self, v: int) -> None:
        self.alive[v] = False

    def restore(self, v: int) -> None:
        self.alive[v] = True

    def clone(self) -> "MemoryGraph":
        g = MemoryGraph(self.n)
        g.adj = [d.copy() for d in self.adj]
        g.alive = self.alive.copy()
        g._orig_comp = self._orig_comp
        return g

    # ---- original connectivity (fixed, computed once) -----------------
    def original_components(self) -> np.ndarray:
        """Component id per node in the *undeleted* graph.

        Cached because it is the denominator of every retention metric: how
        many relationships existed among the nodes a policy chose to keep.
        """
        if self._orig_comp is not None:
            return self._orig_comp
        comp = np.full(self.n, -1, dtype=np.int32)
        cid = 0
        for start in range(self.n):
            if comp[start] != -1:
                continue
            stack = [start]
            comp[start] = cid
            while stack:
                v = stack.pop()
                for w in self.adj[v]:
                    if comp[w] == -1:
                        comp[w] = cid
                        stack.append(w)
            cid += 1
        self._orig_comp = comp
        return comp


def build_graph(
    store: MemoryStore,
    *,
    target_degree: float = 6.0,
    cross_samples: int = 32,
    min_similarity: float = 0.15,
    candidates_per_node: int | None = None,
    seed: int = 0,
) -> MemoryGraph:
    """Build the relationship graph at a requested average degree."""
    n = len(store)
    g = MemoryGraph(n)
    if n < 2:
        return g

    rng = np.random.default_rng(seed)
    k = candidates_per_node or max(2, int(round(target_degree)) + 2)
    vecs = store.vectors

    # Block by category: only compare memories that could plausibly relate.
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(int(store.category[i]), []).append(i)

    cand: dict[tuple[int, int], float] = {}
    best_of: dict[int, tuple[float, int]] = {}

    def offer(i: int, j: int, s: float) -> None:
        if i == j or s < min_similarity:
            return
        key = (i, j) if i < j else (j, i)
        if s > cand.get(key, -1.0):
            cand[key] = s
        if s > best_of.get(i, (-1.0, -1))[0]:
            best_of[i] = (s, j)
        if s > best_of.get(j, (-1.0, -1))[0]:
            best_of[j] = (s, i)

    for members in buckets.values():
        idx = np.asarray(members)
        sub = vecs[idx]
        sims = sub @ sub.T
        np.fill_diagonal(sims, -np.inf)
        take = min(k, len(idx) - 1)
        if take <= 0:
            continue
        top = np.argpartition(-sims, take - 1, axis=1)[:, :take]
        for a in range(len(idx)):
            for b in top[a]:
                offer(int(idx[a]), int(idx[b]), float(sims[a, b]))

    # A fixed sample of cross-category candidates, so bridge memories whose
    # content spans two topics can actually form the edges that make them
    # bridges.  Cost is O(V * cross_samples), not O(V^2).
    if cross_samples > 0 and n > 1:
        picks = rng.integers(0, n, size=(n, cross_samples))
        for i in range(n):
            row = picks[i]
            sims = vecs[row] @ vecs[i]
            for j, s in zip(row, sims):
                offer(i, int(j), float(s))

    # Mutual-nearest-neighbour backbone keeps the graph from being a few hubs
    # plus a cloud of isolated nodes.
    chosen: set[tuple[int, int]] = set()
    for i, (s, j) in best_of.items():
        if j >= 0:
            chosen.add((i, j) if i < j else (j, i))

    budget = int(round(n * target_degree / 2.0))
    if len(chosen) < budget:
        rest = [(s, e) for e, s in cand.items() if e not in chosen]
        rest.sort(key=lambda t: -t[0])
        for _, e in rest[: budget - len(chosen)]:
            chosen.add(e)

    for (u, v) in chosen:
        g.add_edge(u, v, cand[(u, v)])
    return g
