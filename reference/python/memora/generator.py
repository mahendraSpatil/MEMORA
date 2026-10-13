"""Synthetic memory generator with a planted cluster structure.

Why not "same category -> edge": that rule produces near-cliques with almost
no cut vertices, and then every eviction policy scores identically.  A
generator that cannot produce bridges cannot show a structural method
working or failing, so it cannot be used to test one.

This generator plants:

* ``n_categories`` topic clusters, as centroids in a latent space;
* ordinary memories drawn near one centroid;
* ``bridge_fraction`` bridge memories drawn between *two* centroids.

Bridges are the memories CAFA is supposed to protect.  Their number is a
sweep parameter precisely so the experiments can show how much of CAFA's
advantage depends on them existing.
"""

from __future__ import annotations

import numpy as np

from .model import Memory, MemoryStore

TOPICS = [
    "lecture notes", "assignment", "exam paper", "lab report", "meeting notes",
    "screenshot", "invoice", "photo", "chat log", "bookmark", "email thread",
    "code snippet", "reference paper", "todo list", "contact card",
]


def generate_store(
    n: int,
    *,
    n_categories: int = 12,
    bridge_fraction: float = 0.05,
    dim: int = 16,
    cluster_spread: float = 0.55,
    size_log_mu: float = 0.9,
    size_log_sigma: float = 0.9,
    max_age_days: float = 365.0,
    importance_noise: float = 0.0,
    seed: int = 0,
) -> MemoryStore:
    """Build ``n`` synthetic memories.

    ``importance_noise`` is the standard deviation of the gap between what a
    policy can see (``observed_importance``) and what evaluation scores
    (``true_importance``).  At 0.0 the two coincide, which is the optimistic
    case; sweep it upward to test robustness.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(n_categories, dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    n_bridges = int(round(n * bridge_fraction))
    bridge_ids = set(rng.choice(n, size=n_bridges, replace=False).tolist()) if n_bridges else set()

    memories: list[Memory] = []
    for i in range(n):
        cat = int(rng.integers(n_categories))
        if i in bridge_ids and n_categories > 1:
            other = int(rng.integers(n_categories - 1))
            other += int(other >= cat)
            mix = float(rng.uniform(0.35, 0.65))
            base = mix * centroids[cat] + (1.0 - mix) * centroids[other]
        else:
            base = centroids[cat]
        vec = base + cluster_spread * rng.normal(size=dim)
        vec /= np.linalg.norm(vec) or 1.0

        size_mb = float(np.clip(rng.lognormal(size_log_mu, size_log_sigma), 0.1, 400.0))
        true_imp = float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0))
        obs_imp = (
            float(np.clip(true_imp + rng.normal(0.0, importance_noise), 0.0, 1.0))
            if importance_noise > 0 else true_imp
        )
        age = float(rng.uniform(0.0, max_age_days))
        # Recent access is more likely on young, important memories.
        recency = float(age * rng.beta(1.0, 1.0 + 3.0 * true_imp))
        lam = 1.0 + 25.0 * true_imp * np.exp(-recency / 60.0)
        access = int(rng.poisson(lam))

        memories.append(
            Memory(
                id=i,
                name=f"{TOPICS[cat % len(TOPICS)]} #{i}",
                category=cat,
                size_mb=size_mb,
                true_importance=true_imp,
                observed_importance=obs_imp,
                age_days=age,
                recency_days=recency,
                access_count=access,
                vector=vec,
                is_bridge=i in bridge_ids,
            )
        )
    return MemoryStore(memories)


def apply_importance_degree_correlation(
    store: MemoryStore, degrees: np.ndarray, rho: float, seed: int = 0
) -> None:
    """Re-rank importance so it correlates with node degree at strength ``rho``.

    If important memories are already the well-connected ones, LRU and LFU do
    accidentally well and CAFA has little left to add; at negative rho the
    opposite holds.  Sweeping this is what separates "CAFA works" from "this
    dataset happened to be kind".
    """
    if abs(rho) < 1e-9:
        return
    rng = np.random.default_rng(seed)
    n = len(store)
    deg_rank = _rank_uniform(degrees.astype(np.float64))
    noise_rank = _rank_uniform(rng.normal(size=n))
    blended = rho * deg_rank + np.sqrt(max(0.0, 1.0 - rho * rho)) * noise_rank
    order = np.argsort(np.argsort(blended))
    sorted_importance = np.sort(store.true_importance)
    new_true = sorted_importance[order]
    delta = store.observed_importance - store.true_importance
    store.true_importance[:] = new_true
    store.observed_importance[:] = np.clip(new_true + delta, 0.0, 1.0)
    for i, m in enumerate(store.memories):
        m.true_importance = float(new_true[i])
        m.observed_importance = float(store.observed_importance[i])


def _rank_uniform(a: np.ndarray) -> np.ndarray:
    n = len(a)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return (ranks / (n - 1)) * 2.0 - 1.0
