#!/usr/bin/env python3
"""Figures for the report and the poster, straight from results/*.csv.

Nothing is recomputed here. The C++ benchmark owns every number; this only
draws them. That separation is deliberate -- plotting code is the part most
likely to be rewritten the night before, and it must not be able to change a
result.

    make results     # ./bin/memora --all
    make figures     # this script

Needs only matplotlib. No pandas, no seaborn.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    sys.exit("matplotlib is required: pip install matplotlib")

RESULTS = Path("results")
FIGURES = Path("figures")

# Weakest to strongest, so legends read in a sensible order.
ORDER = ["random", "fifo", "lru", "lfu", "size_greedy",
         "value_greedy", "degree_cafa", "cafa_static", "cafa_local", "cafa_repair"]
COLOR = {
    "random": "#9aa0a6", "fifo": "#b0bec5", "lru": "#78909c", "lfu": "#546e7a",
    "size_greedy": "#c98a3a", "value_greedy": "#d05a2a", "degree_cafa": "#6a8f3c",
    "cafa_static": "#3d7ea6", "cafa_local": "#1f4e79", "cafa_repair": "#6b3fa0",
}
HIGHLIGHT = {"value_greedy", "cafa_local", "cafa_repair"}


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        print(f"  skip {name}: {path} not found")
        return []
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("policy", "transform"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def agg(rows, xkey, ykey, policies=None):
    """mean and 95% CI of ykey by (policy, xkey)."""
    buckets = defaultdict(list)
    for r in rows:
        if policies and r["policy"] not in policies:
            continue
        buckets[(r["policy"], r[xkey])].append(r[ykey])
    out = defaultdict(list)
    for (pol, x), vals in buckets.items():
        n = len(vals)
        mean = sum(vals) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in vals) / (n - 1)
            ci = 1.96 * math.sqrt(var / n)
        else:
            ci = 0.0
        out[pol].append((x, mean, ci))
    for pol in out:
        out[pol].sort()
    return out


def ordered(keys):
    return sorted(keys, key=lambda p: ORDER.index(p) if p in ORDER else 99)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------- figure 1
def fig_density():
    """Poster finding 1 and 3: the headline gap, and where it disappears."""
    rows = load("density")
    if not rows:
        return
    data = agg(rows, "actual_degree", "pairwise_retention")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pol in ordered(data):
        xs, ys, cis = zip(*data[pol])
        lw = 2.2 if pol in HIGHLIGHT else 1.1
        ax.errorbar(xs, ys, yerr=cis, label=pol, color=COLOR.get(pol), linewidth=lw,
                    marker="o", markersize=3.5, capsize=2, alpha=1.0 if pol in HIGHLIGHT else 0.7)
    style(ax, "Context preserved vs graph density (equal bytes freed, 25% budget)",
          "average degree", "pairwise connectivity retention")
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    save(fig, "fig1_density.png")


# --------------------------------------------------------------- figure 2
def fig_ablation():
    """Poster finding 2: what the graph is actually buying, by density."""
    rows = load("density")
    if not rows:
        return
    data = agg(rows, "actual_degree", "pairwise_retention",
               policies={"value_greedy", "degree_cafa", "cafa_local", "cafa_repair"})
    base = dict((x, m) for x, m, _ in data.get("value_greedy", []))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pol in ordered([p for p in data if p != "value_greedy"]):
        xs = [x for x, _, _ in data[pol] if x in base]
        ys = [100 * (m - base[x]) for x, m, _ in data[pol] if x in base]
        ax.plot(xs, ys, label=f"{pol} - value_greedy", color=COLOR.get(pol),
                linewidth=2.0, marker="o", markersize=4)
    ax.axhline(0, color="#d05a2a", linewidth=1.4, linestyle="--")
    ax.text(ax.get_xlim()[1], 0.4, "value_greedy (CAFA with w_C = 0)", ha="right",
            va="bottom", fontsize=8, color="#d05a2a")
    style(ax, "Ablation: points of context retention added by the graph term",
          "average degree", "percentage points vs the no-graph ablation")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig2_ablation.png")


# --------------------------------------------------------------- figure 3
def fig_pareto():
    """The trade-off nobody can argue with: importance kept vs context kept."""
    rows = load("main")
    if not rows:
        return
    pts = defaultdict(lambda: [[], []])
    for r in rows:
        pts[r["policy"]][0].append(r["importance_retained"])
        pts[r["policy"]][1].append(r["pairwise_retention"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for pol in ordered(pts):
        xs, ys = pts[pol]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        big = pol in HIGHLIGHT
        ax.scatter([mx], [my], s=110 if big else 55, color=COLOR.get(pol),
                   edgecolor="white", linewidth=1.2, zorder=3 if big else 2)
        ax.annotate(pol, (mx, my), textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=COLOR.get(pol),
                    fontweight="bold" if big else "normal")
    style(ax, "Both objectives at once (mean of 10 seeds, equal bytes freed)",
          "importance retained (true, held out)", "pairwise connectivity retention")
    save(fig, "fig3_pareto.png")


# --------------------------------------------------------------- figure 4
def fig_weight():
    """Why these weights: the Pareto curve as w_C is swept."""
    rows = load("weight")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for pol in ordered({r["policy"] for r in rows}):
        ctx = agg(rows, "w_context", "pairwise_retention", policies={pol})[pol]
        imp = dict((x, m) for x, m, _ in
                   agg(rows, "w_context", "importance_retained", policies={pol})[pol])
        xs = [x for x, _, _ in ctx]
        ax.plot(xs, [m for _, m, _ in ctx], marker="o", markersize=4,
                color=COLOR.get(pol), linewidth=2, label=f"{pol}: context")
        ax.plot(xs, [imp[x] for x in xs], marker="s", markersize=3.5, linestyle="--",
                color=COLOR.get(pol), linewidth=1.2, alpha=0.7,
                label=f"{pol}: importance")
    style(ax, "Context weight sweep: what w_C buys and what it costs",
          "w_C (context weight)", "retention")
    ax.legend(fontsize=7.5, frameon=False)
    save(fig, "fig4_weight.png")


# --------------------------------------------------------------- figure 5
def fig_runtime():
    """The price of structural awareness, in milliseconds."""
    rows = load("scale")
    if not rows:
        return
    data = agg(rows, "n", "runtime_s",
               policies={"lru", "value_greedy", "cafa_static", "cafa_local", "cafa_repair"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for pol in ordered(data):
        xs, ys, _ = zip(*data[pol])
        axes[0].plot(xs, [y * 1000 for y in ys], marker="o", markersize=4,
                     color=COLOR.get(pol), linewidth=2, label=pol)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    style(axes[0], "Eviction time vs store size", "memories (V)", "milliseconds")
    axes[0].legend(fontsize=8, frameon=False)

    build = agg(rows, "n", "graph_build_s", policies={"cafa_local"})
    evict = data.get("cafa_local", [])
    if build.get("cafa_local") and evict:
        xs = [x for x, _, _ in build["cafa_local"]]
        axes[1].plot(xs, [m * 1000 for _, m, _ in build["cafa_local"]], marker="o",
                     color="#b0451b", linewidth=2, label="graph construction")
        axes[1].plot(xs, [m * 1000 for _, m, _ in evict], marker="o",
                     color=COLOR["cafa_local"], linewidth=2, label="CAFA eviction")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        # Bucketed construction beats a naive O(V^2) build by ~20x, which is
        # enough that the incremental Tarjan, not the build, is what costs at
        # V >= 1000. For every other policy the build still dominates.
        style(axes[1], "Where the time actually goes", "memories (V)", "milliseconds")
        axes[1].legend(fontsize=8, frameon=False)
    save(fig, "fig5_runtime.png")


# --------------------------------------------------------------- figure 6
def fig_budget():
    rows = load("budget")
    if not rows:
        return
    data = agg(rows, "budget_frac", "context_query_success")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for pol in ordered(data):
        xs, ys, cis = zip(*data[pol])
        lw = 2.2 if pol in HIGHLIGHT else 1.1
        ax.errorbar([x * 100 for x in xs], ys, yerr=cis, label=pol, color=COLOR.get(pol),
                    linewidth=lw, marker="o", markersize=3.5, capsize=2,
                    alpha=1.0 if pol in HIGHLIGHT else 0.7)
    style(ax, "Held-out context queries still answerable after eviction",
          "bytes freed (% of store)", "context query success")
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    save(fig, "fig6_budget.png")


# --------------------------------------------------------------- figure 7
def fig_cutweight():
    """Does Tarjan earn its keep over plain degree, and where?"""
    rows = load("cutweight")
    if not rows:
        return
    by_deg = defaultdict(list)
    for r in rows:
        by_deg[round(r["target_degree"])].append(r)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    shades = ["#9ec5e8", "#3d7ea6", "#12354f"]
    for i, deg in enumerate(sorted(by_deg)):
        data = agg(by_deg[deg], "cut_weight", "pairwise_retention")
        for pol, series in data.items():
            xs, ys, cis = zip(*series)
            ax.errorbar(xs, ys, yerr=cis, marker="o", markersize=4, capsize=2,
                        color=shades[i % len(shades)], linewidth=2,
                        label=f"avg degree {deg}")
    ax.text(0.01, 0.02, "0 = degree only", transform=ax.transAxes, fontsize=8,
            color="#666")
    ax.text(0.99, 0.02, "1 = Tarjan damage only", transform=ax.transAxes, fontsize=8,
            color="#666", ha="right")
    style(ax, "Structural signal: Tarjan damage vs weighted degree",
          "cut_weight", "pairwise connectivity retention")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "fig7_cutweight.png")


def summary():
    """The three numbers to put on the poster."""
    rows = load("density")
    if not rows:
        return
    print("\nHeadline numbers (pairwise connectivity retention, 25% budget):")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[round(r["target_degree"])][r["policy"]].append(r["pairwise_retention"])
    print(f"  {'degree':>7} {'lru':>9} {'value_greedy':>14} {'cafa_local':>12} "
          f"{'cafa_repair':>13} {'ablation gap':>14}")
    for deg in sorted(by):
        m = {p: sum(v) / len(v) for p, v in by[deg].items()}
        gap = m.get("cafa_local", 0) - m.get("value_greedy", 0)
        print(f"  {deg:>7} {m.get('lru', 0):>9.3f} {m.get('value_greedy', 0):>14.3f} "
              f"{m.get('cafa_local', 0):>12.3f} {m.get('cafa_repair', 0):>13.3f} "
              f"{gap:>+14.3f}")

    knap = load("knapsack")
    if knap:
        gaps = defaultdict(list)
        for r in knap:
            gaps[r["policy"]].append(r["gap_pct"])
        print("\nGreedy vs exact DP optimum (V = 120):")
        for pol, g in gaps.items():
            print(f"  {pol:>14}: {sum(g) / len(g):.2f}% above optimal "
                  f"(worst {max(g):.2f}%)")


def main() -> int:
    if not RESULTS.exists():
        sys.exit("no results/ directory -- run `make results` first")
    for fn in (fig_density, fig_ablation, fig_pareto, fig_weight, fig_runtime,
               fig_budget, fig_cutweight):
        fn()
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
