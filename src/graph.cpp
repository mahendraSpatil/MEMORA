#include "memora/graph.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>

namespace memora {

void MemoryGraph::add_edge(int u, int v, double w) {
    if (u == v) return;
    auto& au = adj_[static_cast<std::size_t>(u)];
    for (auto& e : au) {
        if (e.to == v) {
            e.w = w;
            for (auto& f : adj_[static_cast<std::size_t>(v)])
                if (f.to == u) f.w = w;
            return;
        }
    }
    au.push_back({v, w});
    adj_[static_cast<std::size_t>(v)].push_back({u, w});
    orig_comp_.clear();
}

void MemoryGraph::sort_adjacency() {
    for (auto& row : adj_)
        std::sort(row.begin(), row.end(),
                  [](const Edge& a, const Edge& b) { return a.to < b.to; });
}

bool MemoryGraph::has_edge(int u, int v) const {
    for (const auto& e : adj_[static_cast<std::size_t>(u)])
        if (e.to == v) return true;
    return false;
}

int MemoryGraph::degree(int v) const {
    int d = 0;
    for (const auto& e : adj_[static_cast<std::size_t>(v)])
        if (alive(e.to)) ++d;
    return d;
}

double MemoryGraph::weighted_degree(int v) const {
    double d = 0.0;
    for (const auto& e : adj_[static_cast<std::size_t>(v)])
        if (alive(e.to)) d += e.w;
    return d;
}

std::vector<int> MemoryGraph::neighbours(int v) const {
    std::vector<int> out;
    out.reserve(adj_[static_cast<std::size_t>(v)].size());
    for (const auto& e : adj_[static_cast<std::size_t>(v)])
        if (alive(e.to)) out.push_back(e.to);
    return out;
}

int MemoryGraph::live_count() const {
    int c = 0;
    for (auto a : alive_) c += (a != 0);
    return c;
}

long long MemoryGraph::edge_count() const {
    long long c = 0;
    for (int v = 0; v < n_; ++v) {
        if (!alive(v)) continue;
        for (const auto& e : adj_[static_cast<std::size_t>(v)])
            if (alive(e.to) && e.to > v) ++c;
    }
    return c;
}

const std::vector<int>& MemoryGraph::original_components() const {
    if (!orig_comp_.empty()) return orig_comp_;
    orig_comp_.assign(static_cast<std::size_t>(n_), -1);
    std::vector<int> stack;
    int cid = 0;
    for (int start = 0; start < n_; ++start) {
        if (orig_comp_[static_cast<std::size_t>(start)] != -1) continue;
        stack.clear();
        stack.push_back(start);
        orig_comp_[static_cast<std::size_t>(start)] = cid;
        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            for (const auto& e : adj_[static_cast<std::size_t>(v)]) {
                if (orig_comp_[static_cast<std::size_t>(e.to)] == -1) {
                    orig_comp_[static_cast<std::size_t>(e.to)] = cid;
                    stack.push_back(e.to);
                }
            }
        }
        ++cid;
    }
    return orig_comp_;
}

void connected_components(const MemoryGraph& g, std::vector<int>& comp_out,
                          std::vector<int>& sizes_out) {
    const int n = g.n();
    comp_out.assign(static_cast<std::size_t>(n), -1);
    sizes_out.clear();
    std::vector<int> stack;
    int cid = 0;
    for (int start = 0; start < n; ++start) {
        if (!g.alive(start) || comp_out[static_cast<std::size_t>(start)] != -1) continue;
        stack.clear();
        stack.push_back(start);
        comp_out[static_cast<std::size_t>(start)] = cid;
        int count = 0;
        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            ++count;
            for (const auto& e : g.adj(v)) {
                if (g.alive(e.to) && comp_out[static_cast<std::size_t>(e.to)] == -1) {
                    comp_out[static_cast<std::size_t>(e.to)] = cid;
                    stack.push_back(e.to);
                }
            }
        }
        sizes_out.push_back(count);
        ++cid;
    }
}

// ---------------------------------------------------------------- builder
namespace {

struct PairKey {
    int a, b;
    bool operator==(const PairKey& o) const { return a == o.a && b == o.b; }
};

struct PairHash {
    std::size_t operator()(const PairKey& k) const noexcept {
        return (static_cast<std::size_t>(k.a) << 32) ^ static_cast<std::size_t>(k.b);
    }
};

}  // namespace

MemoryGraph build_graph(const Store& store, const GraphParams& p) {
    const int n = static_cast<int>(store.size());
    MemoryGraph g(n);
    if (n < 2) return g;

    std::mt19937_64 rng(p.seed ^ 0x9E3779B97F4A7C15ULL);
    const int k = p.candidates_per_node > 0
                      ? p.candidates_per_node
                      : std::max(2, int(std::lround(p.target_degree)) + 2);

    std::unordered_map<PairKey, double, PairHash> cand;
    cand.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * 2);
    std::vector<double> best_sim(static_cast<std::size_t>(n), -1.0);
    std::vector<int> best_of(static_cast<std::size_t>(n), -1);

    auto offer = [&](int i, int j, double s) {
        if (i == j || s < p.min_similarity) return;
        PairKey key{std::min(i, j), std::max(i, j)};
        auto it = cand.find(key);
        if (it == cand.end() || s > it->second) cand[key] = s;
        if (s > best_sim[static_cast<std::size_t>(i)]) {
            best_sim[static_cast<std::size_t>(i)] = s;
            best_of[static_cast<std::size_t>(i)] = j;
        }
        if (s > best_sim[static_cast<std::size_t>(j)]) {
            best_sim[static_cast<std::size_t>(j)] = s;
            best_of[static_cast<std::size_t>(j)] = i;
        }
    };

    // Block by category: only compare memories that could plausibly relate.
    std::unordered_map<int, std::vector<int>> buckets;
    for (int i = 0; i < n; ++i) buckets[store[i].category].push_back(i);

    std::vector<std::pair<double, int>> scored;
    for (const auto& [cat, members] : buckets) {
        (void)cat;
        const int m = static_cast<int>(members.size());
        if (m < 2) continue;
        const int take = std::min(k, m - 1);
        for (int a = 0; a < m; ++a) {
            scored.clear();
            scored.reserve(static_cast<std::size_t>(m - 1));
            for (int b = 0; b < m; ++b) {
                if (a == b) continue;
                scored.emplace_back(store.similarity(members[a], members[b]), members[b]);
            }
            std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                              [](const auto& x, const auto& y) { return x.first > y.first; });
            for (int t = 0; t < take; ++t) offer(members[a], scored[t].second, scored[t].first);
        }
    }

    // A fixed sample of cross-category candidates, so bridge memories whose
    // content spans two topics can actually form the edges that make them
    // bridges. Cost is O(V * cross_samples), not O(V^2).
    if (p.cross_samples > 0) {
        std::uniform_int_distribution<int> pick(0, n - 1);
        for (int i = 0; i < n; ++i)
            for (int s = 0; s < p.cross_samples; ++s) {
                int j = pick(rng);
                offer(i, j, store.similarity(i, j));
            }
    }

    // Mutual-nearest-neighbour backbone keeps the graph from being a few hubs
    // plus a cloud of isolated nodes.
    std::unordered_map<PairKey, double, PairHash> chosen;
    chosen.reserve(cand.size());
    for (int i = 0; i < n; ++i) {
        int j = best_of[static_cast<std::size_t>(i)];
        if (j < 0) continue;
        PairKey key{std::min(i, j), std::max(i, j)};
        chosen[key] = cand[key];
    }

    const std::size_t budget =
        static_cast<std::size_t>(std::llround(double(n) * p.target_degree / 2.0));
    if (chosen.size() < budget) {
        std::vector<std::pair<double, PairKey>> rest;
        rest.reserve(cand.size());
        for (const auto& [key, s] : cand)
            if (!chosen.count(key)) rest.emplace_back(s, key);
        const std::size_t need = budget - chosen.size();
        if (rest.size() > need)
            std::partial_sort(rest.begin(), rest.begin() + need, rest.end(),
                              [](const auto& x, const auto& y) { return x.first > y.first; });
        else
            std::sort(rest.begin(), rest.end(),
                      [](const auto& x, const auto& y) { return x.first > y.first; });
        for (std::size_t i = 0; i < std::min(need, rest.size()); ++i)
            chosen[rest[i].second] = rest[i].first;
    }

    for (const auto& [key, w] : chosen) g.add_edge(key.a, key.b, w);
    g.sort_adjacency();
    return g;
}

}  // namespace memora
