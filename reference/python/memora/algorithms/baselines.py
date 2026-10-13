"""Baseline eviction policies.

Two of these are not classical, and they are the important ones.

``size_greedy`` shows how much of CAFA's advantage is just "delete big
things".  ``value_greedy`` (in :mod:`memora.algorithms.cafa`, as CAFA with
``w_C = 0``) is the ablation: the identical policy with the graph switched
off.  Without those two, comparing CAFA against FIFO/LRU/LFU only shows that
reading importance beats ignoring it, which nobody doubts and which has
nothing to do with the graph.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from ..graph import MemoryGraph
from ..model import MemoryStore
from .base import EvictionResult, evict_in_order


# ---------------------------------------------------------------- FIFO
def fifo(store: MemoryStore, graph: MemoryGraph, required_mb: float, **_) -> EvictionResult:
    """Oldest first.  A queue ordered by creation time; O(V log V) to build."""
    t0 = time.perf_counter()
    order = np.argsort(-store.age_days, kind="mergesort")
    return evict_in_order("fifo", store, graph, required_mb, order, t0)


# ----------------------------------------------------------------- LRU
class _Node:
    __slots__ = ("key", "prev", "next")

    def __init__(self, key: int):
        self.key = key
        self.prev: "_Node | None" = None
        self.next: "_Node | None" = None


class LRUList:
    """Doubly linked list + hashmap: O(1) touch, O(1) evict from the tail.

    The benchmark evicts from a static snapshot, so nothing is touched during
    a run, but ``touch`` is implemented because the O(1) move-to-front is the
    reason LRU uses this structure at all and the report needs to show it.
    """

    def __init__(self):
        self.head: _Node | None = None
        self.tail: _Node | None = None
        self.index: dict[int, _Node] = {}

    def push_front(self, key: int) -> None:
        node = _Node(key)
        self.index[key] = node
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node

    def _unlink(self, node: _Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.prev = node.next = None

    def touch(self, key: int) -> None:
        node = self.index.get(key)
        if node is None or node is self.head:
            return
        self._unlink(node)
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node

    def pop_lru(self) -> int | None:
        if self.tail is None:
            return None
        node = self.tail
        self._unlink(node)
        del self.index[node.key]
        return node.key


def lru(store: MemoryStore, graph: MemoryGraph, required_mb: float, **_) -> EvictionResult:
    t0 = time.perf_counter()
    lst = LRUList()
    # Build most-recently-used first, so the tail is the least recent.
    for key in np.argsort(store.recency_days, kind="mergesort"):
        lst.push_front(int(key))
    freed = 0.0
    deleted: list[int] = []
    while freed < required_mb:
        key = lst.pop_lru()
        if key is None:
            break
        graph.delete(key)
        deleted.append(key)
        freed += float(store.size_mb[key])
    return EvictionResult("lru", deleted, freed, required_mb, time.perf_counter() - t0, graph)


# ----------------------------------------------------------------- LFU
def lfu(store: MemoryStore, graph: MemoryGraph, required_mb: float, **_) -> EvictionResult:
    """Least frequently accessed first, via a min-heap on access count."""
    t0 = time.perf_counter()
    heap = [(int(store.access_count[i]), int(i)) for i in range(len(store))]
    heapq.heapify(heap)
    freed = 0.0
    deleted: list[int] = []
    while freed < required_mb and heap:
        _, v = heapq.heappop(heap)
        graph.delete(v)
        deleted.append(v)
        freed += float(store.size_mb[v])
    return EvictionResult("lfu", deleted, freed, required_mb, time.perf_counter() - t0, graph)


# -------------------------------------------------------------- others
def random_policy(
    store: MemoryStore, graph: MemoryGraph, required_mb: float, *, seed: int = 0, **_
) -> EvictionResult:
    """The floor.  Without it the other numbers have no scale."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(store))
    return evict_in_order("random", store, graph, required_mb, order, t0)


def size_greedy(store: MemoryStore, graph: MemoryGraph, required_mb: float, **_) -> EvictionResult:
    """Largest first: meets the budget in the fewest deletions, ignores value."""
    t0 = time.perf_counter()
    order = np.argsort(-store.size_mb, kind="mergesort")
    return evict_in_order("size_greedy", store, graph, required_mb, order, t0)
