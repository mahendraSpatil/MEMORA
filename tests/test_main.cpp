// Validation suite. No external framework: one binary, plain assertions.
//
// The checks that actually catch bugs are the ones with an independent
// oracle, so Tarjan is verified against a brute-force recomputation of
// connectivity, and the knapsack DP against exhaustive search. "It compiled
// and produced plausible numbers" is not a test.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "memora/generator.hpp"
#include "memora/graph.hpp"
#include "memora/knapsack.hpp"
#include "memora/metrics.hpp"
#include "memora/policies.hpp"
#include "memora/scoring.hpp"
#include "memora/tarjan.hpp"

using namespace memora;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::cout << "  FAIL: " << what << "\n";
    }
}

void section(const std::string& name) { std::cout << "== " << name << "\n"; }

// --- independent oracle: components of (component of v) minus v -----------
std::vector<int> pieces_after_removing(const MemoryGraph& g, int v) {
    const int n = g.n();
    std::vector<int> comp(static_cast<std::size_t>(n), -1);
    // Restrict to v's own component.
    std::vector<int> stack{v};
    std::vector<char> in_comp(static_cast<std::size_t>(n), 0);
    in_comp[static_cast<std::size_t>(v)] = 1;
    while (!stack.empty()) {
        const int x = stack.back();
        stack.pop_back();
        for (const auto& e : g.adj(x))
            if (g.alive(e.to) && !in_comp[static_cast<std::size_t>(e.to)]) {
                in_comp[static_cast<std::size_t>(e.to)] = 1;
                stack.push_back(e.to);
            }
    }
    std::vector<int> sizes;
    int cid = 0;
    for (int s = 0; s < n; ++s) {
        if (s == v || !in_comp[static_cast<std::size_t>(s)] || !g.alive(s)) continue;
        if (comp[static_cast<std::size_t>(s)] != -1) continue;
        stack.assign(1, s);
        comp[static_cast<std::size_t>(s)] = cid;
        int count = 0;
        while (!stack.empty()) {
            const int x = stack.back();
            stack.pop_back();
            ++count;
            for (const auto& e : g.adj(x)) {
                if (e.to == v || !g.alive(e.to)) continue;
                if (comp[static_cast<std::size_t>(e.to)] == -1) {
                    comp[static_cast<std::size_t>(e.to)] = cid;
                    stack.push_back(e.to);
                }
            }
        }
        sizes.push_back(count);
        ++cid;
    }
    return sizes;
}

int component_size(const MemoryGraph& g, int v) {
    std::vector<char> seen(static_cast<std::size_t>(g.n()), 0);
    std::vector<int> stack{v};
    seen[static_cast<std::size_t>(v)] = 1;
    int count = 0;
    while (!stack.empty()) {
        const int x = stack.back();
        stack.pop_back();
        ++count;
        for (const auto& e : g.adj(x))
            if (g.alive(e.to) && !seen[static_cast<std::size_t>(e.to)]) {
                seen[static_cast<std::size_t>(e.to)] = 1;
                stack.push_back(e.to);
            }
    }
    return count;
}

MemoryGraph random_graph(int n, double p, std::mt19937_64& rng) {
    MemoryGraph g(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (u(rng) < p) g.add_edge(i, j, 1.0);
    g.sort_adjacency();
    return g;
}

// --- 1. Tarjan vs brute force --------------------------------------------
void test_tarjan() {
    section("Tarjan: articulation points and pair damage vs brute force");
    std::mt19937_64 rng(12345);
    int graphs = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const int n = 2 + int(rng() % 39);
        const double p = std::vector<double>{0.04, 0.08, 0.15, 0.30, 0.50}[rng() % 5];
        MemoryGraph g = random_graph(n, p, rng);
        ++graphs;

        StructuralIndex idx(g);
        idx.rebuild();

        for (int v = 0; v < n; ++v) {
            const std::vector<int> pieces = pieces_after_removing(g, v);
            const long long nc = component_size(g, v);
            const long long denom = (nc - 1) * (nc - 2) / 2;
            long long kept = 0;
            for (int c : pieces) kept += static_cast<long long>(c) * (c - 1) / 2;
            const double expected = double(denom - kept);

            if (std::abs(idx.damage()[static_cast<std::size_t>(v)] - expected) > 1e-9) {
                check(false, "damage mismatch at n=" + std::to_string(n) +
                                 " v=" + std::to_string(v) + " got " +
                                 std::to_string(idx.damage()[static_cast<std::size_t>(v)]) +
                                 " want " + std::to_string(expected));
                return;
            }
            // The invariant the whole design leans on.
            const bool is_ap = pieces.size() >= 2;
            const bool says_ap = idx.damage()[static_cast<std::size_t>(v)] > 0.0;
            if (is_ap != says_ap) {
                check(false, "articulation flag mismatch at v=" + std::to_string(v));
                return;
            }
        }
    }
    check(true, "tarjan");
    std::cout << "  " << graphs << " random graphs, damage and AP flag exact\n";
}

// --- 2. incremental update == full rebuild -------------------------------
void test_incremental_matches_full() {
    section("StructuralIndex: local update equals a full rebuild");
    std::mt19937_64 rng(999);
    for (int trial = 0; trial < 60; ++trial) {
        const int n = 20 + int(rng() % 40);
        MemoryGraph g = random_graph(n, 0.08, rng);
        StructuralIndex incremental(g);
        incremental.rebuild();

        for (int step = 0; step < 8; ++step) {
            int v = -1;
            for (int t = 0; t < 50; ++t) {
                const int c = int(rng() % static_cast<unsigned>(n));
                if (g.alive(c)) { v = c; break; }
            }
            if (v < 0) break;
            g.kill(v);
            incremental.update_after_delete(v);

            StructuralIndex fresh(g);
            fresh.rebuild();
            for (int u = 0; u < n; ++u) {
                if (!g.alive(u)) continue;
                if (std::abs(incremental.damage()[static_cast<std::size_t>(u)] -
                             fresh.damage()[static_cast<std::size_t>(u)]) > 1e-9) {
                    check(false, "incremental damage drifted at u=" + std::to_string(u));
                    return;
                }
            }
        }
    }
    check(true, "incremental update");
    std::cout << "  60 graphs x 8 deletions, incremental damage identical to rebuild\n";
}

// --- 3. knapsack DP vs exhaustive ----------------------------------------
void test_knapsack() {
    section("Knapsack: O(V*B) DP vs exhaustive search");
    std::mt19937_64 rng(4242);
    std::uniform_real_distribution<double> val(0.0, 1.0);
    for (int trial = 0; trial < 60; ++trial) {
        const int n = 6 + int(rng() % 7);  // <= 12 items
        std::vector<double> sizes(static_cast<std::size_t>(n));
        std::vector<double> values(static_cast<std::size_t>(n));
        double total = 0.0;
        for (int i = 0; i < n; ++i) {
            sizes[static_cast<std::size_t>(i)] = double(1 + rng() % 15);
            values[static_cast<std::size_t>(i)] = val(rng);
            total += sizes[static_cast<std::size_t>(i)];
        }
        const double required = std::floor(total * 0.35);
        const double dp = optimal_value_lost(sizes, values, required, 1.0);
        const double bf = brute_force_value_lost(sizes, values, required);
        if (std::abs(dp - bf) > 1e-6) {
            check(false, "DP " + std::to_string(dp) + " != brute force " + std::to_string(bf));
            return;
        }
    }
    check(true, "knapsack DP");
    std::cout << "  60 instances, DP matches exhaustive search exactly\n";
}

// --- 4. every policy meets the budget ------------------------------------
void test_budget_invariant() {
    section("Policies: budget met, overshoot bounded, repair never hurts");
    GeneratorParams gp;
    gp.n = 400;
    gp.seed = 7;
    Store store = generate_store(gp);
    GraphParams ggp;
    ggp.target_degree = 4.0;
    ggp.seed = 7;
    MemoryGraph graph = build_graph(store, ggp);

    const std::vector<double> sp = static_features(store).combined;
    for (double frac : {0.05, 0.25, 0.5}) {
        const double required = frac * store.total_size_mb();
        for (const auto& [name, fn] : policy_registry()) {
            PolicyParams pp;
            pp.static_part = &sp;
            MemoryGraph working = graph;
            const EvictionResult r = fn(store, working, required, pp);

            check(r.bytes_freed >= required - 1e-6,
                  name + " freed " + std::to_string(r.bytes_freed) + " < required " +
                      std::to_string(required));

            // Deleted list and alive mask must agree.
            int dead = 0;
            for (int v = 0; v < working.n(); ++v)
                if (!working.alive(v)) ++dead;
            check(dead == r.n_deleted(), name + " deleted list disagrees with alive mask");

            double sum = 0.0;
            for (int v : r.deleted) sum += store[v].size_mb;
            check(std::abs(sum - r.bytes_freed) < 1e-6, name + " bytes_freed inconsistent");
        }
    }

    // The repair pass must never increase the value lost.
    const double required = 0.25 * store.total_size_mb();
    PolicyParams with_repair;
    with_repair.static_part = &sp;
    PolicyParams without = with_repair;
    without.restore_overshoot = false;

    MemoryGraph a = graph, b = graph;
    const EvictionResult ra = cafa(store, a, required, with_repair, "with");
    const EvictionResult rb = cafa(store, b, required, without, "without");
    check(ra.bytes_freed <= rb.bytes_freed + 1e-9, "repair pass increased bytes freed");
    check(ra.bytes_freed >= required - 1e-6, "repair pass broke the budget");
    check(ra.n_deleted() <= rb.n_deleted(), "repair pass deleted more items");
    std::cout << "  repair restored " << ra.restored << " memories, overshoot "
              << rb.overshoot() << " -> " << ra.overshoot() << "\n";
}

// --- 5. the ablation must actually differ --------------------------------
void test_ablation_has_an_effect() {
    section("Ablation: w_C = 0 and w_C > 0 must make different decisions");
    GeneratorParams gp;
    gp.n = 800;
    gp.seed = 3;
    Store store = generate_store(gp);
    GraphParams ggp;
    ggp.target_degree = 3.0;  // sparse: where structure matters most
    ggp.seed = 3;
    MemoryGraph graph = build_graph(store, ggp);

    const std::vector<double> sp = static_features(store).combined;
    const double required = 0.25 * store.total_size_mb();
    const std::vector<QueryPair> queries = sample_context_queries(graph, 300, 3, 3);

    PolicyParams pp;
    pp.static_part = &sp;
    MemoryGraph a = graph, b = graph;
    const EvictionResult abl = policy_value_greedy(store, a, required, pp);
    const EvictionResult full = policy_cafa_local(store, b, required, pp);

    std::set<int> sa(abl.deleted.begin(), abl.deleted.end());
    std::set<int> sb(full.deleted.begin(), full.deleted.end());
    check(sa != sb, "CAFA and its ablation deleted the identical set -- w_C has no effect");

    const Metrics ma = evaluate(store, graph, a, abl, queries);
    const Metrics mb = evaluate(store, graph, b, full, queries);
    std::cout << "  pairwise retention: value_greedy " << ma.pairwise_retention
              << "  cafa_local " << mb.pairwise_retention << "\n";
    check(mb.pairwise_retention > ma.pairwise_retention,
          "CAFA did not beat its own ablation on a sparse graph");
}

// --- 6. metrics sanity ----------------------------------------------------
void test_metrics_bounds() {
    section("Metrics: bounds and the no-op case");
    GeneratorParams gp;
    gp.n = 300;
    gp.seed = 11;
    Store store = generate_store(gp);
    GraphParams ggp;
    ggp.target_degree = 4.0;
    ggp.seed = 11;
    MemoryGraph graph = build_graph(store, ggp);
    const std::vector<QueryPair> queries = sample_context_queries(graph, 200, 3, 11);

    // Deleting nothing must score a perfect 1.0 everywhere.
    MemoryGraph untouched = graph;
    EvictionResult nothing;
    nothing.policy = "none";
    nothing.required_mb = 0.0;
    const Metrics m = evaluate(store, graph, untouched, nothing, queries);
    check(std::abs(m.pairwise_retention - 1.0) < 1e-9, "empty eviction != 1.0 pairwise");
    check(std::abs(m.neighbourhood_retention - 1.0) < 1e-9, "empty eviction != 1.0 nbr");
    check(std::abs(m.importance_retained - 1.0) < 1e-9, "empty eviction != 1.0 importance");
    check(std::abs(m.context_query_success - 1.0) < 1e-9, "empty eviction != 1.0 queries");

    const std::vector<double> sp = static_features(store).combined;
    PolicyParams pp;
    pp.static_part = &sp;
    for (const auto& [name, fn] : policy_registry()) {
        MemoryGraph w = graph;
        const EvictionResult r = fn(store, w, 0.3 * store.total_size_mb(), pp);
        const Metrics mm = evaluate(store, graph, w, r, queries);
        check(mm.pairwise_retention >= -1e-9 && mm.pairwise_retention <= 1.0 + 1e-9,
              name + " pairwise retention out of [0,1]");
        check(mm.importance_retained >= -1e-9 && mm.importance_retained <= 1.0 + 1e-9,
              name + " importance retained out of [0,1]");
    }
}

// --- 7. helpers -----------------------------------------------------------
void test_rank_normalize() {
    section("rank_normalize: bounds, ties, monotonicity");
    const std::vector<double> a = {5.0, 1.0, 3.0, 3.0, 9.0};
    const std::vector<double> r = rank_normalize(a);
    check(std::abs(r[1] - 0.0) < 1e-12, "minimum should map to 0");
    check(std::abs(r[4] - 1.0) < 1e-12, "maximum should map to 1");
    check(std::abs(r[2] - r[3]) < 1e-12, "ties should share a rank");
    check(r[2] < r[0] && r[0] < r[4], "ranks should preserve order");
    check(rank_normalize({}).empty(), "empty input");
    check(rank_normalize({7.0}).size() == 1, "single element");
}

void test_lru_list() {
    section("LRUList: O(1) touch and tail eviction");
    LRUList l;
    for (int k : {0, 1, 2, 3}) l.push_front(k);  // head = 3, tail = 0
    check(l.pop_lru() == 0, "tail should be the least recent");
    l.touch(1);                                   // 1 becomes most recent
    check(l.pop_lru() == 2, "after touching 1, tail should be 2");
    check(l.pop_lru() == 3, "then 3");
    check(l.pop_lru() == 1, "touched item evicted last");
    check(l.pop_lru() == -1, "empty list returns -1");
}

void test_determinism() {
    section("Determinism: identical seed, identical eviction");
    GeneratorParams gp;
    gp.n = 500;
    gp.seed = 21;
    GraphParams ggp;
    ggp.target_degree = 4.0;
    ggp.seed = 21;

    std::vector<int> first;
    for (int run = 0; run < 2; ++run) {
        Store store = generate_store(gp);
        MemoryGraph graph = build_graph(store, ggp);
        const std::vector<double> sp = static_features(store).combined;
        PolicyParams pp;
        pp.static_part = &sp;
        MemoryGraph w = graph;
        const EvictionResult r =
            policy_cafa_local(store, w, 0.25 * store.total_size_mb(), pp);
        if (run == 0) first = r.deleted;
        else check(first == r.deleted, "two identical runs evicted different memories");
    }
}

}  // namespace

int main() {
    std::cout << "Memora validation suite\n\n";
    test_tarjan();
    test_incremental_matches_full();
    test_knapsack();
    test_budget_invariant();
    test_ablation_has_an_effect();
    test_metrics_bounds();
    test_rank_normalize();
    test_lru_list();
    test_determinism();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    if (g_failures) {
        std::cout << g_failures << " FAILURES\n";
        return 1;
    }
    std::cout << "all good\n";
    return 0;
}
