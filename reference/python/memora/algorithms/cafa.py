"""CAFA -- the Context-Aware Forgetting Algorithm, and its ablations.

Evict in increasing value per byte until the space requirement is met, where
the value of a memory includes how much of the graph's connectivity its
removal would destroy.

Three variants, because the difference between them is a result rather than
an implementation detail:

``static``
    Score once, then delete K items from the heap.  This is what the original
    proposal's pseudocode does, and it is wrong in an interesting way: after
    a few deletions the scores describe a graph that no longer exists.  Kept
    as the speed reference and to measure what staleness costs.

``local``
    After each deletion, re-run Tarjan on the affected component only.  The
    default.  Components fragment as the run proceeds, so the average cost
    per update falls away quickly.

``local`` + ``bridge_repair`` (CAFA-R)
    Also reconnect the deleted memory's surviving neighbours to each other
    where they are similar enough.  The relationship survives its
    intermediary.  Costs O(deg(v)^2) per deletion and usually moves context
    preservation more than any amount of weight tuning.

Re-scoring neighbours after each deletion is also what prevents *cascading
cluster erasure*: delete a low-value peripheral memory and its neighbour
becomes peripheral too, so the greedy walks inward and wipes a whole cluster.
``static`` cannot see that happening; ``local`` can.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from ..graph import MemoryGraph
from ..model import MemoryStore
from ..scoring import (
    retention_value, static_features, structural_score, value_density,
)
from ..tarjan import StructuralIndex
from .base import EvictionResult, repair_overshoot

MAX_BRIDGE_DEGREE = 12  # keeps the O(deg^2) repair bounded


def cafa(
    store: MemoryStore,
    graph: MemoryGraph,
    required_mb: float,
    *,
    w_context: float = 0.30,
    mode: str = "local",
    cut_weight: float = 0.7,
    transform: str = "sqrt",
    bridge_repair: bool = False,
    bridge_threshold: float = 0.25,
    restore_overshoot: bool = True,
    static_part: np.ndarray | None = None,
    name: str | None = None,
    **_,
) -> EvictionResult:
    t0 = time.perf_counter()
    n = len(store)
    sizes = store.size_mb
    static_part = static_features(store).combined if static_part is None else static_part
    label = name or f"cafa_{mode}" + ("_repair" if bridge_repair else "")

    uses_graph = w_context > 0.0 or bridge_repair
    needs_tarjan = uses_graph and cut_weight > 0.0
    index: StructuralIndex | None = None
    wdeg = np.zeros(n, dtype=np.float64)
    damage_scale = 1.0
    max_wdeg = 1.0

    if uses_graph:
        for v in range(n):
            if graph.alive[v]:
                wdeg[v] = graph.weighted_degree(v)
        max_wdeg = float(wdeg.max()) or 1.0
        if needs_tarjan:
            index = StructuralIndex(graph)
            index.rebuild()
            # Calibrate against the worst cut actually present at t=0 and then
            # hold that scale fixed, so scores do not drift under us.
            damage_scale = max(1.0, float(index.damage.max()))
            raw_damage = index.damage
        else:
            raw_damage = np.zeros(n, dtype=np.float64)
        context = structural_score(
            raw_damage, wdeg, damage_scale, max_wdeg,
            cut_weight=cut_weight, transform=transform,
        )
    else:
        context = np.zeros(n, dtype=np.float64)

    value = retention_value(static_part, context, w_context)
    rho = value_density(value, sizes)

    version = np.zeros(n, dtype=np.int64)
    heap = [(float(rho[v]), 0, int(v)) for v in range(n) if graph.alive[v]]
    heapq.heapify(heap)

    freed = 0.0
    deleted: list[int] = []
    value_lost: dict[int, float] = {}
    bridges_added = 0

    while freed < required_mb and heap:
        _, ver, v = heapq.heappop(heap)
        if not graph.alive[v] or ver != version[v]:
            continue  # lazy deletion: heapq has no decrease-key
        graph.delete(v)
        deleted.append(v)
        value_lost[v] = float(value[v])
        freed += float(sizes[v])

        if mode != "local" or not uses_graph:
            continue

        neigh = graph.neighbours(v)
        if bridge_repair and len(neigh) > 1:
            ranked = sorted(neigh, key=lambda u: -graph.adj[v].get(u, 0.0))
            ranked = ranked[:MAX_BRIDGE_DEGREE]
            for a_i in range(len(ranked)):
                a = ranked[a_i]
                for b in ranked[a_i + 1:]:
                    if b in graph.adj[a]:
                        continue
                    s = store.similarity(a, b)
                    if s >= bridge_threshold:
                        graph.add_edge(a, b, s)
                        bridges_added += 1

        # Only the component v was in can have changed, so re-run Tarjan on
        # that alone.  With cut_weight 0 there is no structure to refresh and
        # only the immediate neighbours' degrees move.
        if index is not None:
            changed = index.update_after_delete(v)
            damage_now = index.damage
        else:
            changed = neigh
            damage_now = np.zeros(n, dtype=np.float64)
        if not changed:
            continue
        idx = np.fromiter(changed, dtype=np.int64, count=len(changed))
        for u in changed:
            wdeg[u] = graph.weighted_degree(u)
        context[idx] = structural_score(
            damage_now[idx], wdeg[idx], damage_scale, max_wdeg,
            cut_weight=cut_weight, transform=transform,
        )
        value[idx] = retention_value(static_part[idx], context[idx], w_context)
        rho[idx] = value_density(value[idx], sizes[idx])
        for u in changed:
            version[u] += 1
            heapq.heappush(heap, (float(rho[u]), int(version[u]), int(u)))

    restored = 0
    if restore_overshoot:
        deleted, freed, restored = repair_overshoot(
            store, graph, deleted, value_lost, freed, required_mb
        )

    return EvictionResult(
        policy=label,
        deleted=deleted,
        bytes_freed=freed,
        required_mb=required_mb,
        runtime_s=time.perf_counter() - t0,
        graph=graph,
        stats={
            "restored": restored,
            "bridges_added": bridges_added,
            "analysis_calls": index.analysis_calls if index else 0,
            "nodes_visited": index.nodes_visited if index else 0,
            "w_context": w_context,
            "mode": mode,
        },
    )


# ------------------------------------------------------------- variants
def value_greedy(store, graph, required_mb, **kw) -> EvictionResult:
    """**The ablation.**  CAFA with w_C = 0: identical policy, no graph.

    The headline claim of this project is the gap between this row and the
    CAFA row.  Everything else in the results table is context.
    """
    kw.pop("w_context", None)
    kw.pop("mode", None)
    return cafa(store, graph, required_mb, w_context=0.0, mode="static",
                name="value_greedy", **kw)


def degree_cafa(store, graph, required_mb, *, w_context: float = 0.30, **kw) -> EvictionResult:
    """Graph signal without Tarjan: protect high weighted-degree nodes.

    If this gets most of CAFA's benefit at O(1) per node, say so -- that is a
    more interesting finding than pretending otherwise.
    """
    kw.pop("cut_weight", None)
    kw.pop("mode", None)
    return cafa(store, graph, required_mb, w_context=w_context, mode="local",
                cut_weight=0.0, name="degree_cafa", **kw)


def cafa_static(store, graph, required_mb, **kw) -> EvictionResult:
    kw.pop("mode", None)
    return cafa(store, graph, required_mb, mode="static", name="cafa_static", **kw)


def cafa_local(store, graph, required_mb, **kw) -> EvictionResult:
    kw.pop("mode", None)
    return cafa(store, graph, required_mb, mode="local", name="cafa_local", **kw)


def cafa_repair(store, graph, required_mb, **kw) -> EvictionResult:
    kw.pop("mode", None)
    kw.pop("bridge_repair", None)
    return cafa(store, graph, required_mb, mode="local", bridge_repair=True,
                name="cafa_repair", **kw)
