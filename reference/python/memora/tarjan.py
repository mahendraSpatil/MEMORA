"""Iterative Tarjan with graded pair-connectivity damage.

An articulation point flag is a yes/no answer, and a yes/no answer cannot
rank.  Removing a cut vertex that splits off one leaf costs one connected
pair; removing one that halves a 1,000-node cluster costs about 250,000.  A
fixed "articulation bonus" treats those as the same event, which is how a
context-aware policy ends up making visibly silly choices.

So this module computes, in the *same* DFS that finds the cut vertices, how
many connected pairs each removal would destroy:

    D(v) = C(n-1, 2) - sum_i C(c_i, 2)

where ``c_i`` are the sizes of the components left behind when v is removed
from its component of size n.  Subtree sizes fall out of the DFS for free, so
this is still O(V+E) and still one pass.

Useful invariant, and a test in :mod:`memora.validate`:
``D(v) > 0`` exactly when v is an articulation point.

The DFS is iterative.  A recursive one dies on a long chain at V=10,000:
Python's default limit is 1,000 frames, and raising it tends to segfault
rather than raise.
"""

from __future__ import annotations

import numpy as np

from .graph import MemoryGraph


class StructuralIndex:
    """Maintains per-node structural damage and component membership.

    ``rebuild()`` does the whole live graph in O(V+E).
    ``update_after_delete(v)`` re-analyses only the component v was in, which
    is what makes CAFA-Local affordable: after the first few deletions the
    components fragment and the cost per update falls away quickly.
    """

    __slots__ = (
        "g", "n", "damage", "comp_of", "comp_size",
        "_disc", "_low", "_sub", "_seen", "_parent", "_gen", "_next_cid",
        "analysis_calls", "nodes_visited",
    )

    def __init__(self, graph: MemoryGraph):
        self.g = graph
        n = graph.n
        self.n = n
        self.damage = np.zeros(n, dtype=np.float64)
        self.comp_of = np.full(n, -1, dtype=np.int32)
        self.comp_size: dict[int, int] = {}
        self._disc = np.zeros(n, dtype=np.int64)
        self._low = np.zeros(n, dtype=np.int64)
        self._sub = np.zeros(n, dtype=np.int64)
        self._seen = np.zeros(n, dtype=np.int64)
        self._parent = np.full(n, -1, dtype=np.int64)
        self._gen = 0
        self._next_cid = 0
        self.analysis_calls = 0
        self.nodes_visited = 0

    # -- public ---------------------------------------------------------
    def rebuild(self) -> list[int]:
        """Analyse every live node.  O(V + E)."""
        self.comp_size.clear()
        self._next_cid = 0
        self.comp_of.fill(-1)
        self.damage.fill(0.0)
        return self._analyse(np.flatnonzero(self.g.alive).tolist())

    def update_after_delete(self, v: int) -> list[int]:
        """Re-analyse only the component that ``v`` was part of.

        Call *after* ``graph.delete(v)``.  Returns the nodes whose damage
        changed, so the caller can re-push exactly those into its heap.
        """
        self.damage[v] = 0.0
        self.comp_of[v] = -1
        seeds = self.g.neighbours(v)
        if not seeds:
            return []
        return self._analyse(seeds)

    def articulation_points(self) -> set[int]:
        return {int(v) for v in np.flatnonzero((self.damage > 0) & self.g.alive)}

    # -- internals ------------------------------------------------------
    def _analyse(self, roots: list[int]) -> list[int]:
        self.analysis_calls += 1
        self._gen += 1
        gen = self._gen
        alive = self.g.alive
        seen = self._seen
        visited: list[int] = []
        for root in roots:
            if not alive[root] or seen[root] == gen:
                continue
            order, pieces = self._dfs(int(root), gen)
            cid = self._next_cid
            self._next_cid += 1
            n_comp = len(order)
            self.comp_size[cid] = n_comp
            denom = (n_comp - 1) * (n_comp - 2) // 2
            for v in order:
                self.comp_of[v] = cid
                ps = pieces.get(v)
                if ps:
                    kept = 0
                    total = 0
                    for c in ps:
                        kept += c * (c - 1) // 2
                        total += c
                    rest = n_comp - 1 - total
                    if rest > 0:
                        kept += rest * (rest - 1) // 2
                    self.damage[v] = float(denom - kept)
                else:
                    self.damage[v] = 0.0
            visited.extend(order)
        self.nodes_visited += len(visited)
        return visited

    def _dfs(self, root: int, gen: int):
        adj = self.g.adj
        alive = self.g.alive
        disc, low, sub = self._disc, self._low, self._sub
        seen, parent = self._seen, self._parent

        pieces: dict[int, list[int]] = {}
        order: list[int] = [root]
        timer = 0
        seen[root] = gen
        disc[root] = low[root] = timer
        timer += 1
        sub[root] = 1
        parent[root] = -1
        stack = [(root, iter(adj[root]))]

        while stack:
            v, it = stack[-1]
            descended = False
            for w in it:
                if not alive[w]:
                    continue
                if seen[w] != gen:
                    seen[w] = gen
                    disc[w] = low[w] = timer
                    timer += 1
                    sub[w] = 1
                    parent[w] = v
                    order.append(w)
                    stack.append((w, iter(adj[w])))
                    descended = True
                    break
                if w != parent[v] and disc[w] < low[v]:
                    low[v] = disc[w]
            if descended:
                continue
            stack.pop()
            p = parent[v]
            if p != -1:
                sub[p] += sub[v]
                if low[v] < low[p]:
                    low[p] = low[v]
                if low[v] >= disc[p]:
                    pieces.setdefault(int(p), []).append(int(sub[v]))
        return order, pieces


def connected_components(graph: MemoryGraph) -> tuple[np.ndarray, dict[int, int]]:
    """Component id per live node plus component sizes, on the live subgraph."""
    comp = np.full(graph.n, -1, dtype=np.int32)
    sizes: dict[int, int] = {}
    alive = graph.alive
    cid = 0
    for start in range(graph.n):
        if not alive[start] or comp[start] != -1:
            continue
        stack = [start]
        comp[start] = cid
        count = 0
        while stack:
            v = stack.pop()
            count += 1
            for w in graph.adj[v]:
                if alive[w] and comp[w] == -1:
                    comp[w] = cid
                    stack.append(w)
        sizes[cid] = count
        cid += 1
    return comp, sizes
