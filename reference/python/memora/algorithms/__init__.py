"""Policy registry.

``POLICIES`` is the single list the benchmark iterates.  Adding a policy here
is all it takes for it to appear in every figure.
"""

from __future__ import annotations

from .base import EvictionResult, evict_in_order, repair_overshoot
from .baselines import LRUList, fifo, lfu, lru, random_policy, size_greedy
from .cafa import cafa, cafa_local, cafa_repair, cafa_static, degree_cafa, value_greedy

POLICIES = {
    "random": random_policy,
    "fifo": fifo,
    "lru": lru,
    "lfu": lfu,
    "size_greedy": size_greedy,
    "value_greedy": value_greedy,   # the ablation: CAFA with w_C = 0
    "degree_cafa": degree_cafa,
    "cafa_static": cafa_static,
    "cafa_local": cafa_local,
    "cafa_repair": cafa_repair,
}

#: Order used in tables and figures, weakest to strongest.
POLICY_ORDER = [
    "random", "fifo", "lru", "lfu", "size_greedy",
    "value_greedy", "degree_cafa", "cafa_static", "cafa_local", "cafa_repair",
]

__all__ = [
    "POLICIES", "POLICY_ORDER", "EvictionResult", "evict_in_order",
    "repair_overshoot", "LRUList", "fifo", "lfu", "lru", "random_policy",
    "size_greedy", "cafa", "cafa_local", "cafa_repair", "cafa_static",
    "degree_cafa", "value_greedy",
]
