"""Memory records and the in-memory store.

A Memory is deliberately plain data: every policy in :mod:`memora.algorithms`
reads the same fields, so no policy can quietly use information another one
cannot see.

Note the split between ``true_importance`` and ``observed_importance``.
Eviction policies may only read the *observed* value; evaluation scores the
*true* one.  Without that split a policy that reads importance is graded on
the very number it was handed, and the comparison proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Memory:
    """One stored item."""

    id: int
    name: str
    category: int
    size_mb: float
    true_importance: float          # ground truth, evaluation only
    observed_importance: float      # noisy view, this is what policies see
    age_days: float                 # days since creation
    recency_days: float             # days since last access
    access_count: int
    vector: np.ndarray = field(repr=False, default=None)
    is_bridge: bool = False

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Memory(id={self.id}, {self.name!r}, cat={self.category}, "
            f"{self.size_mb:.1f}MB, imp={self.true_importance:.2f})"
        )


class MemoryStore:
    """A fixed collection of memories, indexed 0..n-1.

    Ids are contiguous integers on purpose.  String keys and dict-of-list
    adjacency cost 2-4x in the Tarjan inner loop at V=10,000 for no benefit;
    display names live in a side table instead.
    """

    __slots__ = (
        "memories", "size_mb", "true_importance", "observed_importance",
        "age_days", "recency_days", "access_count", "category", "vectors",
    )

    def __init__(self, memories: list[Memory]):
        self.memories = memories
        n = len(memories)
        self.size_mb = np.empty(n, dtype=np.float64)
        self.true_importance = np.empty(n, dtype=np.float64)
        self.observed_importance = np.empty(n, dtype=np.float64)
        self.age_days = np.empty(n, dtype=np.float64)
        self.recency_days = np.empty(n, dtype=np.float64)
        self.access_count = np.empty(n, dtype=np.float64)
        self.category = np.empty(n, dtype=np.int32)
        for i, m in enumerate(memories):
            if m.id != i:
                raise ValueError(f"memory ids must be 0..n-1 in order (got {m.id} at {i})")
            self.size_mb[i] = m.size_mb
            self.true_importance[i] = m.true_importance
            self.observed_importance[i] = m.observed_importance
            self.age_days[i] = m.age_days
            self.recency_days[i] = m.recency_days
            self.access_count[i] = m.access_count
            self.category[i] = m.category
        self.vectors = np.vstack([m.vector for m in memories]) if n else np.zeros((0, 0))

    def __len__(self) -> int:
        return len(self.memories)

    def __getitem__(self, i: int) -> Memory:
        return self.memories[i]

    @property
    def total_size_mb(self) -> float:
        return float(self.size_mb.sum())

    def similarity(self, i: int, j: int) -> float:
        """Cosine similarity between two memories' latent vectors."""
        a, b = self.vectors[i], self.vectors[j]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
