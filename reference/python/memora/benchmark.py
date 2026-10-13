"""Benchmark harness: one CSV row per (policy, configuration, seed).

Every chart, every table and the dashboard read this CSV.  Metrics are never
computed inside the UI, because that couples the results to the part of the
project most likely to be unfinished the night before.

Graph construction time is reported separately from eviction time.  It is not
part of the policy being compared, and folding it in would swamp the
differences between policies.

Usage::

    python -m memora.benchmark --sweep main     --out results/main.csv
    python -m memora.benchmark --sweep density  --out results/density.csv
    python -m memora.benchmark --list
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .algorithms import POLICIES, POLICY_ORDER
from .generator import apply_importance_degree_correlation, generate_store
from .graph import build_graph
from .knapsack import optimal_value_lost
from .metrics import evaluate, sample_context_queries
from .scoring import static_features


@dataclass(frozen=True)
class Config:
    n: int = 1500
    n_categories: int = 12
    target_degree: float = 4.0
    bridge_fraction: float = 0.05
    budget_frac: float = 0.25
    importance_noise: float = 0.0
    importance_degree_rho: float = 0.0
    w_context: float = 0.30
    cut_weight: float = 0.70
    transform: str = "sqrt"
    n_queries: int = 500
    seed: int = 0


def run_config(cfg: Config, policies: list[str] | None = None) -> list[dict]:
    """Build one dataset and run every policy against the same byte budget."""
    policies = policies or POLICY_ORDER

    store = generate_store(
        cfg.n,
        n_categories=cfg.n_categories,
        bridge_fraction=cfg.bridge_fraction,
        importance_noise=cfg.importance_noise,
        seed=cfg.seed,
    )
    t0 = time.perf_counter()
    graph = build_graph(store, target_degree=cfg.target_degree, seed=cfg.seed)
    build_s = time.perf_counter() - t0

    if cfg.importance_degree_rho:
        degrees = np.array([graph.degree(v) for v in range(graph.n)], dtype=np.float64)
        apply_importance_degree_correlation(
            store, degrees, cfg.importance_degree_rho, seed=cfg.seed
        )

    queries = sample_context_queries(graph, n_queries=cfg.n_queries, seed=cfg.seed)
    static_part = static_features(store).combined
    required = cfg.budget_frac * store.total_size_mb

    rows = []
    for name in policies:
        fn = POLICIES[name]
        result = fn(
            store, graph.clone(), required,
            seed=cfg.seed,
            static_part=static_part,
            w_context=cfg.w_context,
            cut_weight=cfg.cut_weight,
            transform=cfg.transform,
        )
        row = evaluate(store, graph, result, queries)
        row.update(asdict(cfg))
        row["graph_build_s"] = round(build_s, 5)
        row["edges"] = graph.edge_count()
        row["actual_degree"] = round(2 * graph.edge_count() / max(graph.n, 1), 3)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ sweeps
def _seeds(k: int) -> list[int]:
    return list(range(k))


SWEEPS: dict[str, dict] = {
    "main": dict(grid={"seed": _seeds(10)}, base=Config()),
    "density": dict(
        grid={"target_degree": [2, 3, 4, 6, 8, 12], "seed": _seeds(10)}, base=Config()
    ),
    "budget": dict(
        grid={"budget_frac": [0.05, 0.10, 0.25, 0.40, 0.50], "seed": _seeds(10)},
        base=Config(),
    ),
    "weight": dict(
        grid={"w_context": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "seed": _seeds(6)},
        base=Config(),
        policies=["cafa_local", "cafa_repair"],
    ),
    "cutweight": dict(
        grid={"cut_weight": [0.0, 0.25, 0.5, 0.7, 0.9, 1.0],
              "target_degree": [2, 4, 8], "seed": _seeds(5)},
        base=Config(),
        policies=["cafa_local"],
    ),
    "noise": dict(
        grid={"importance_noise": [0.0, 0.1, 0.2, 0.3], "seed": _seeds(6)}, base=Config()
    ),
    "correlation": dict(
        grid={"importance_degree_rho": [-0.5, 0.0, 0.5], "seed": _seeds(6)}, base=Config()
    ),
    "bridges": dict(
        grid={"bridge_fraction": [0.0, 0.01, 0.05, 0.10], "seed": _seeds(6)}, base=Config()
    ),
    "scale": dict(
        grid={"n": [100, 500, 1000, 2000, 5000], "seed": _seeds(3)},
        base=Config(n_queries=200),
    ),
}


def run_sweep(name: str, out: Path, quiet: bool = False) -> list[dict]:
    spec = SWEEPS[name]
    base: Config = spec["base"]
    grid: dict = spec["grid"]
    policies = spec.get("policies")
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, combo in enumerate(combos, 1):
        cfg = replace(base, **dict(zip(keys, combo)))
        rows.extend(run_config(cfg, policies))
        if not quiet:
            done = time.perf_counter() - t0
            print(f"  [{i}/{len(combos)}] {dict(zip(keys, combo))}  ({done:.1f}s)", flush=True)
    write_csv(rows, out)
    return rows


def knapsack_gap(sizes_n: int = 120, seeds: int = 8, budget_frac: float = 0.25) -> list[dict]:
    """How far from optimal is the greedy?  Exact DP on small instances."""
    from .algorithms.cafa import cafa

    rows = []
    for seed in range(seeds):
        cfg = Config(n=sizes_n, seed=seed, n_categories=6, n_queries=100)
        store = generate_store(cfg.n, n_categories=cfg.n_categories, seed=seed)
        graph = build_graph(store, target_degree=cfg.target_degree, seed=seed)
        static_part = static_features(store).combined
        required = cfg.budget_frac * store.total_size_mb
        for policy, kw in [
            ("value_greedy", dict(w_context=0.0, mode="static")),
            ("cafa_local", dict(w_context=cfg.w_context, mode="local")),
        ]:
            g = graph.clone()
            res = cafa(store, g, required, static_part=static_part,
                       cut_weight=cfg.cut_weight, name=policy, **kw)
            # Value function the policy actually optimised, for a like-for-like
            # comparison against the exact optimum of that same objective.
            value = static_part if kw["w_context"] == 0 else None
            if value is None:
                from .scoring import retention_value, structural_score
                from .tarjan import StructuralIndex
                idx = StructuralIndex(graph.clone())
                idx.rebuild()
                wdeg = np.array([graph.weighted_degree(v) for v in range(graph.n)])
                ctx = structural_score(
                    idx.damage, wdeg, max(1.0, float(idx.damage.max())),
                    float(wdeg.max()) or 1.0, cut_weight=cfg.cut_weight)
                value = retention_value(static_part, ctx, cfg.w_context)
            greedy_lost = float(value[res.deleted].sum())
            opt = optimal_value_lost(store.size_mb, value, required, scale=1.0)
            rows.append({
                "seed": seed, "n": cfg.n, "policy": policy,
                "greedy_value_lost": round(greedy_lost, 5),
                "optimal_value_lost": round(opt, 5),
                "gap_pct": round(100.0 * (greedy_lost - opt) / max(opt, 1e-9), 3),
                "n_deleted": res.n_deleted,
                "overshoot": round(res.overshoot, 4),
            })
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", default="main", choices=sorted(SWEEPS) + ["knapsack", "all"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true", help="list sweeps and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for k, v in SWEEPS.items():
            print(f"{k:12s} {v['grid']}")
        return 0

    results = Path("results")
    if args.sweep == "knapsack":
        write_csv(knapsack_gap(), args.out or results / "knapsack.csv")
        return 0
    if args.sweep == "all":
        for name in SWEEPS:
            print(f"== sweep: {name}")
            run_sweep(name, results / f"{name}.csv", quiet=args.quiet)
        write_csv(knapsack_gap(), results / "knapsack.csv")
        return 0

    run_sweep(args.sweep, args.out or results / f"{args.sweep}.csv", quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
