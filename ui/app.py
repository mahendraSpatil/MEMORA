"""Memora web UI — Flask backend.

Exposes the Python reference implementation as JSON API endpoints:
  /api/generate   — create a synthetic memory store + graph
  /api/run        — run selected policies and return metrics + graph state
  /api/step       — step through CAFA one eviction at a time (animated mode)
"""

import json
import sys
import os
import time

# Add the reference Python package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference", "python"))

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

from memora.generator import generate_store
from memora.graph import build_graph, MemoryGraph
from memora.algorithms import POLICIES, POLICY_ORDER
from memora.metrics import evaluate, sample_context_queries
from memora.scoring import static_features, structural_score, retention_value, value_density
from memora.tarjan import StructuralIndex

app = Flask(__name__, static_folder="static", static_url_path="")

# ──────────────────────── session state ────────────────────────
_session = {}


def _graph_to_json(graph: MemoryGraph, store=None, damage=None):
    """Serialize the live portion of a MemoryGraph to JSON-friendly dicts."""
    nodes = []
    for v in range(graph.n):
        node = {"id": v, "alive": bool(graph.alive[v])}
        if store:
            m = store[v]
            node.update({
                "name": m.name,
                "category": int(m.category),
                "size_mb": round(float(m.size_mb), 3),
                "true_importance": round(float(m.true_importance), 3),
                "observed_importance": round(float(m.observed_importance), 3),
                "age_days": round(float(m.age_days), 1),
                "recency_days": round(float(m.recency_days), 1),
                "access_count": int(m.access_count),
                "is_bridge": bool(m.is_bridge),
            })
        if damage is not None:
            node["damage"] = round(float(damage[v]), 2)
        nodes.append(node)

    edges = []
    seen = set()
    for v in range(graph.n):
        for w, wt in graph.adj[v].items():
            key = (min(v, w), max(v, w))
            if key not in seen:
                seen.add(key)
                edges.append({
                    "source": key[0],
                    "target": key[1],
                    "weight": round(float(wt), 4),
                })
    return {"nodes": nodes, "edges": edges}


# ──────────────────────── routes ────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate a synthetic memory store and build the relationship graph."""
    body = request.get_json(force=True) or {}
    n = int(body.get("n", 200))
    n = max(10, min(n, 2000))
    n_categories = int(body.get("n_categories", 8))
    bridge_fraction = float(body.get("bridge_fraction", 0.05))
    target_degree = float(body.get("target_degree", 4.0))
    seed = int(body.get("seed", 42))

    store = generate_store(
        n,
        n_categories=n_categories,
        bridge_fraction=bridge_fraction,
        seed=seed,
    )
    graph = build_graph(store, target_degree=target_degree, seed=seed)

    # Compute initial structural analysis
    si = StructuralIndex(graph)
    si.rebuild()

    _session["store"] = store
    _session["graph_original"] = graph
    _session["seed"] = seed
    _session["target_degree"] = target_degree

    graph_json = _graph_to_json(graph, store, damage=si.damage)
    return jsonify({
        "n": n,
        "total_size_mb": round(float(store.total_size_mb), 2),
        "edge_count": graph.edge_count(),
        "avg_degree": round(2 * graph.edge_count() / n, 2),
        "graph": graph_json,
        "categories": n_categories,
        "bridges": int(sum(1 for m in store.memories if m.is_bridge)),
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run selected eviction policies and return comparative metrics."""
    if "store" not in _session:
        return jsonify({"error": "Generate data first"}), 400

    body = request.get_json(force=True) or {}
    budget_pct = float(body.get("budget_pct", 25.0))
    selected_policies = body.get("policies", POLICY_ORDER)

    store = _session["store"]
    original = _session["graph_original"]
    required_mb = store.total_size_mb * budget_pct / 100.0

    queries = sample_context_queries(original, seed=_session["seed"])

    results = []
    graphs_after = {}
    for pname in POLICY_ORDER:
        if pname not in selected_policies:
            continue
        fn = POLICIES.get(pname)
        if fn is None:
            continue
        g = original.clone()
        t0 = time.perf_counter()
        result = fn(store, g, required_mb, seed=_session["seed"])
        metrics = evaluate(store, original, result, queries)
        metrics["runtime_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Build the after-graph for the last CAFA variant
        if pname in ("cafa_local", "cafa_repair"):
            si = StructuralIndex(g)
            si.rebuild()
            graphs_after[pname] = _graph_to_json(g, store, damage=si.damage)
        else:
            graphs_after[pname] = _graph_to_json(g, store)

        metrics["deleted_ids"] = result.deleted
        results.append(metrics)

    return jsonify({
        "required_mb": round(required_mb, 2),
        "budget_pct": budget_pct,
        "results": results,
        "graphs_after": graphs_after,
    })


@app.route("/api/step", methods=["POST"])
def api_step():
    """Initialize or advance a step-by-step CAFA eviction for animation."""
    if "store" not in _session:
        return jsonify({"error": "Generate data first"}), 400

    body = request.get_json(force=True) or {}
    action = body.get("action", "init")
    budget_pct = float(body.get("budget_pct", 25.0))

    store = _session["store"]
    original = _session["graph_original"]
    required_mb = store.total_size_mb * budget_pct / 100.0

    if action == "init":
        graph = original.clone()
        si = StructuralIndex(graph)
        si.rebuild()

        sf = static_features(store)
        static_part = sf.combined
        sizes = store.size_mb
        n = len(store)

        damage_scale = max(1.0, float(si.damage.max()))
        wdeg = np.array([graph.weighted_degree(v) if graph.alive[v] else 0.0 for v in range(n)])
        max_wdeg = float(wdeg.max()) or 1.0

        context = structural_score(si.damage, wdeg, damage_scale, max_wdeg)
        value = retention_value(static_part, context, 0.30)
        rho = value_density(value, sizes)

        _session["step_graph"] = graph
        _session["step_si"] = si
        _session["step_static"] = static_part
        _session["step_damage_scale"] = damage_scale
        _session["step_max_wdeg"] = max_wdeg
        _session["step_value"] = value
        _session["step_rho"] = rho
        _session["step_freed"] = 0.0
        _session["step_deleted"] = []
        _session["step_required"] = required_mb

        graph_json = _graph_to_json(graph, store, damage=si.damage)
        # Add per-node value/rho
        for nd in graph_json["nodes"]:
            v = nd["id"]
            nd["value"] = round(float(value[v]), 4)
            nd["rho"] = round(float(rho[v]), 4)

        return jsonify({
            "status": "initialized",
            "required_mb": round(required_mb, 2),
            "total_size_mb": round(float(store.total_size_mb), 2),
            "freed": 0.0,
            "n_deleted": 0,
            "graph": graph_json,
        })

    elif action == "next":
        graph = _session.get("step_graph")
        si = _session.get("step_si")
        if graph is None or si is None:
            return jsonify({"error": "Initialize stepper first"}), 400

        freed = _session["step_freed"]
        if freed >= _session["step_required"]:
            return jsonify({"status": "complete", "freed": round(freed, 3), "n_deleted": len(_session["step_deleted"])})

        value = _session["step_value"]
        rho = _session["step_rho"]
        sizes = store.size_mb
        n = len(store)

        # Find the live node with lowest rho
        best_v, best_rho = -1, float("inf")
        for v in range(n):
            if graph.alive[v] and float(rho[v]) < best_rho:
                best_rho = float(rho[v])
                best_v = v

        if best_v < 0:
            return jsonify({"status": "complete", "freed": round(freed, 3), "n_deleted": len(_session["step_deleted"])})

        # Record which neighbors exist before deletion
        neigh_before = graph.neighbours(best_v)

        graph.delete(best_v)
        freed += float(sizes[best_v])
        _session["step_freed"] = freed
        _session["step_deleted"].append(best_v)

        # Re-run Tarjan on affected component
        changed = si.update_after_delete(best_v)

        # Update scores for changed nodes
        static_part = _session["step_static"]
        damage_scale = _session["step_damage_scale"]
        max_wdeg = _session["step_max_wdeg"]
        if changed:
            idx = np.array(changed, dtype=np.int64)
            wdeg_changed = np.array([graph.weighted_degree(u) for u in changed])
            ctx = structural_score(si.damage[idx], wdeg_changed, damage_scale, max_wdeg)
            val = retention_value(static_part[idx], ctx, 0.30)
            rh = value_density(val, sizes[idx])
            for i, u in enumerate(changed):
                value[u] = val[i]
                rho[u] = rh[i]

        deleted_mem = store[best_v]
        graph_json = _graph_to_json(graph, store, damage=si.damage)
        for nd in graph_json["nodes"]:
            v = nd["id"]
            nd["value"] = round(float(value[v]), 4)
            nd["rho"] = round(float(rho[v]), 4)

        return jsonify({
            "status": "stepped",
            "evicted": {
                "id": best_v,
                "name": deleted_mem.name,
                "category": int(deleted_mem.category),
                "size_mb": round(float(deleted_mem.size_mb), 3),
                "importance": round(float(deleted_mem.true_importance), 3),
                "rho": round(best_rho, 4),
                "is_bridge": bool(deleted_mem.is_bridge),
                "was_articulation": float(si.damage[best_v]) > 0 if best_v < len(si.damage) else False,
                "neighbours_affected": len(neigh_before),
            },
            "freed": round(freed, 3),
            "required_mb": round(_session["step_required"], 2),
            "n_deleted": len(_session["step_deleted"]),
            "pct_done": round(freed / _session["step_required"] * 100, 1),
            "graph": graph_json,
        })

    return jsonify({"error": f"Unknown action: {action}"}), 400


if __name__ == "__main__":
    print("Memora UI running at http://localhost:5000")
    app.run(debug=False, port=5000)
