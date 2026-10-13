#include "memora/metrics.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace memora {

std::vector<QueryPair> sample_context_queries(const MemoryGraph& g, int n_queries,
                                              int max_depth, std::uint64_t seed) {
    std::vector<QueryPair> pairs;
    if (g.n() < 2 || n_queries <= 0) return pairs;
    pairs.reserve(static_cast<std::size_t>(n_queries));

    std::mt19937_64 rng(seed ^ 0x2545F4914F6CDD1DULL);
    std::uniform_int_distribution<int> pick(0, g.n() - 1);

    std::vector<int> depth(static_cast<std::size_t>(g.n()), -1);
    std::vector<int> touched, frontier, next, far;
    int attempts = 0;

    while (static_cast<int>(pairs.size()) < n_queries && attempts < n_queries * 25) {
        ++attempts;
        const int u = pick(rng);

        for (int v : touched) depth[static_cast<std::size_t>(v)] = -1;
        touched.clear();
        far.clear();

        depth[static_cast<std::size_t>(u)] = 0;
        touched.push_back(u);
        frontier.assign(1, u);

        for (int d = 1; d <= max_depth && !frontier.empty(); ++d) {
            next.clear();
            for (int x : frontier) {
                for (const auto& e : g.adj(x)) {
                    if (depth[static_cast<std::size_t>(e.to)] == -1) {
                        depth[static_cast<std::size_t>(e.to)] = d;
                        touched.push_back(e.to);
                        next.push_back(e.to);
                        if (d >= 2) far.push_back(e.to);
                    }
                }
            }
            frontier.swap(next);
        }
        if (!far.empty()) {
            std::uniform_int_distribution<std::size_t> jd(0, far.size() - 1);
            pairs.emplace_back(u, far[jd(rng)]);
        }
    }
    for (int v : touched) depth[static_cast<std::size_t>(v)] = -1;
    return pairs;
}

Metrics evaluate(const Store& store, const MemoryGraph& original, const MemoryGraph& after,
                 const EvictionResult& result, const std::vector<QueryPair>& queries) {
    Metrics m;
    m.policy = result.policy;
    m.n_deleted = result.n_deleted();
    m.bytes_freed_mb = result.bytes_freed;
    m.required_mb = result.required_mb;
    m.overshoot = result.overshoot();
    m.runtime_s = result.runtime_s;
    m.restored = result.restored;
    m.bridges_added = result.bridges_added;
    m.analysis_calls = result.analysis_calls;
    m.nodes_visited = result.nodes_visited;

    const int n = original.n();
    std::vector<int> survivors;
    survivors.reserve(static_cast<std::size_t>(n));
    for (int v = 0; v < n; ++v)
        if (after.alive(v)) survivors.push_back(v);
    m.survivors = static_cast<int>(survivors.size());

    // Denominator: how many pairs of the SURVIVING set were connected in the
    // original graph. Deleting a memory that lay on no path between survivors
    // then costs nothing, which a raw component count cannot express.
    const std::vector<int>& orig_comp = original.original_components();
    std::unordered_map<int, long long> orig_groups;
    for (int v : survivors) ++orig_groups[orig_comp[static_cast<std::size_t>(v)]];

    std::vector<int> new_comp, new_sizes;
    connected_components(after, new_comp, new_sizes);

    long long pairs_before = 0, lcc_before = 0;
    for (const auto& [cid, c] : orig_groups) {
        (void)cid;
        pairs_before += c * (c - 1) / 2;
        lcc_before = std::max(lcc_before, c);
    }
    long long pairs_after = 0, lcc_after = 0;
    for (int c : new_sizes) {
        pairs_after += static_cast<long long>(c) * (c - 1) / 2;
        lcc_after = std::max(lcc_after, static_cast<long long>(c));
    }

    m.pairwise_retention =
        pairs_before > 0 ? double(pairs_after) / double(pairs_before) : 1.0;
    m.lcc_ratio = lcc_before > 0 ? double(lcc_after) / double(lcc_before) : 1.0;
    m.components_after = static_cast<int>(new_sizes.size());
    m.components_before = static_cast<int>(orig_groups.size());

    // Neighbourhood retention uses the ORIGINAL adjacency on both sides, so
    // repair edges added by CAFA-R cannot inflate it.
    double total = 0.0;
    int counted = 0;
    for (int v : survivors) {
        const auto& edges = original.adj(v);
        if (edges.empty()) continue;
        int kept = 0;
        for (const auto& e : edges)
            if (after.alive(e.to)) ++kept;
        total += double(kept) / double(edges.size());
        ++counted;
    }
    m.neighbourhood_retention = counted ? total / counted : 1.0;

    double imp_total = 0.0, imp_kept = 0.0;
    long long bridges_total = 0, bridges_kept = 0;
    for (int v = 0; v < n; ++v) {
        imp_total += store[v].true_importance;
        if (after.alive(v)) imp_kept += store[v].true_importance;
        if (store[v].is_bridge) {
            ++bridges_total;
            if (after.alive(v)) ++bridges_kept;
        }
    }
    m.importance_retained = imp_total > 0.0 ? imp_kept / imp_total : 1.0;
    m.bridge_retention = bridges_total > 0 ? double(bridges_kept) / double(bridges_total) : 1.0;

    if (!queries.empty()) {
        int both_alive = 0, still_linked = 0;
        for (const auto& [u, w] : queries) {
            if (!after.alive(u) || !after.alive(w)) continue;
            ++both_alive;
            const int cu = new_comp[static_cast<std::size_t>(u)];
            if (cu != -1 && cu == new_comp[static_cast<std::size_t>(w)]) ++still_linked;
        }
        m.context_query_success = both_alive ? double(still_linked) / both_alive : 1.0;
        m.context_query_survival = double(still_linked) / double(queries.size());
        m.query_coverage = double(both_alive) / double(queries.size());
    }
    return m;
}

}  // namespace memora
