// CAFA -- the Context-Aware Forgetting Algorithm -- and everything it is
// compared against.
//
// Evict in increasing value per byte until the space requirement is met,
// where the value of a memory includes how much of the graph's connectivity
// its removal would destroy.
//
// Three CAFA variants, because the difference between them is a result rather
// than an implementation detail:
//
//   Static            Score once, then delete K items from the heap. This is
//                     what the original proposal's pseudocode does, and it is
//                     wrong in an interesting way: after a few deletions the
//                     scores describe a graph that no longer exists. Kept as
//                     the speed reference and to measure what staleness costs.
//   Local             Re-run Tarjan on the affected component after each
//                     deletion. The default.
//   Local + repair    Also reconnect the deleted memory's surviving
//                     neighbours where they are similar enough, so the
//                     relationship survives its intermediary. O(deg^2) per
//                     deletion.
//
// Re-scoring neighbours after each deletion is also what prevents cascading
// cluster erasure: delete a low-value peripheral memory and its neighbour
// becomes peripheral too, so the greedy walks inward and wipes a whole
// cluster. Static cannot see that happening; Local can.
#include "memora/policies.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <queue>
#include <random>

#include "memora/tarjan.hpp"

namespace memora {

namespace {

constexpr int kMaxBridgeDegree = 12;  // keeps the O(deg^2) repair bounded

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Delete along a precomputed priority order until the budget is met.
EvictionResult evict_in_order(const std::string& name, const Store& store,
                              MemoryGraph& g, double required_mb,
                              const std::vector<int>& order, double t0) {
    EvictionResult r;
    r.policy = name;
    r.required_mb = required_mb;
    for (int v : order) {
        if (r.bytes_freed >= required_mb) break;
        g.kill(v);
        r.deleted.push_back(v);
        r.bytes_freed += store[v].size_mb;
    }
    r.runtime_s = now_s() - t0;
    return r;
}

// Undo deletions the budget did not need.
//
// Greedy overshoots. Budget 10 MB; X is 6 MB worth 0.30, Y is 10 MB worth
// 0.60, Z is 4 MB worth 0.28. Greedy takes X (density 0.050) then Y (0.060):
// 16 MB freed, 0.90 lost. X and Z free exactly 10 MB for 0.58.
//
// Walking the deleted list in decreasing value and restoring anything that
// fits is O(K log K) and can never make the result worse.
int repair_overshoot(const Store& store, MemoryGraph& g, std::vector<int>& deleted,
                     const std::vector<double>& value_lost, double& freed,
                     double required_mb) {
    std::vector<int> by_value = deleted;
    std::sort(by_value.begin(), by_value.end(), [&](int a, int b) {
        if (value_lost[std::size_t(a)] != value_lost[std::size_t(b)])
            return value_lost[std::size_t(a)] > value_lost[std::size_t(b)];
        return a < b;
    });
    int restored = 0;
    for (int v : by_value) {
        if (freed - store[v].size_mb >= required_mb) {
            g.revive(v);
            freed -= store[v].size_mb;
            ++restored;
        }
    }
    if (restored > 0) {
        std::vector<int> kept;
        kept.reserve(deleted.size());
        for (int v : deleted)
            if (!g.alive(v)) kept.push_back(v);
        deleted.swap(kept);
    }
    return restored;
}

std::vector<int> order_by(std::size_t n, const std::vector<double>& key, bool descending) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const double ka = key[std::size_t(a)], kb = key[std::size_t(b)];
        if (ka != kb) return descending ? ka > kb : ka < kb;
        return a < b;  // deterministic tie-break
    });
    return order;
}

struct HeapEntry {
    double rho;
    long long version;
    int node;
};

// std::priority_queue pops the "largest", so this comparator has to report
// that `a` ranks *below* `b` when a should be popped later -- i.e. it is a
// greater-than test, which turns the queue into a min-heap on rho.
struct HeapCmp {
    bool operator()(const HeapEntry& a, const HeapEntry& b) const {
        if (a.rho != b.rho) return a.rho > b.rho;
        return a.node > b.node;
    }
};

}  // namespace

// ------------------------------------------------------------- LRU's list
void LRUList::push_front(int key) {
    if (static_cast<int>(index_.size()) <= key) index_.resize(std::size_t(key) + 1, -1);
    const int slot = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{key, -1, head_});
    index_[std::size_t(key)] = slot;
    if (head_ != -1) nodes_[std::size_t(head_)].prev = slot;
    head_ = slot;
    if (tail_ == -1) tail_ = slot;
}

void LRUList::unlink(int slot) {
    Node& node = nodes_[std::size_t(slot)];
    if (node.prev != -1) nodes_[std::size_t(node.prev)].next = node.next;
    else head_ = node.next;
    if (node.next != -1) nodes_[std::size_t(node.next)].prev = node.prev;
    else tail_ = node.prev;
    node.prev = node.next = -1;
}

void LRUList::touch(int key) {
    if (key < 0 || key >= static_cast<int>(index_.size())) return;
    const int slot = index_[std::size_t(key)];
    if (slot == -1 || slot == head_) return;
    unlink(slot);
    nodes_[std::size_t(slot)].next = head_;
    if (head_ != -1) nodes_[std::size_t(head_)].prev = slot;
    head_ = slot;
}

int LRUList::pop_lru() {
    if (tail_ == -1) return -1;
    const int slot = tail_;
    const int key = nodes_[std::size_t(slot)].key;
    unlink(slot);
    index_[std::size_t(key)] = -1;
    return key;
}

// -------------------------------------------------------------- baselines
EvictionResult policy_random(const Store& store, MemoryGraph& g, double required_mb,
                             const PolicyParams& p) {
    const double t0 = now_s();
    std::vector<int> order(store.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(p.seed ^ 0xA24BAED4963EE407ULL);
    std::shuffle(order.begin(), order.end(), rng);
    return evict_in_order("random", store, g, required_mb, order, t0);
}

EvictionResult policy_fifo(const Store& store, MemoryGraph& g, double required_mb,
                           const PolicyParams&) {
    const double t0 = now_s();
    return evict_in_order("fifo", store, g, required_mb,
                          order_by(store.size(), store.age_days(), true), t0);
}

EvictionResult policy_lru(const Store& store, MemoryGraph& g, double required_mb,
                          const PolicyParams&) {
    const double t0 = now_s();
    LRUList list;
    // Build most-recently-used first, so the tail is the least recent.
    for (int v : order_by(store.size(), store.recency_days(), false)) list.push_front(v);

    EvictionResult r;
    r.policy = "lru";
    r.required_mb = required_mb;
    while (r.bytes_freed < required_mb) {
        const int key = list.pop_lru();
        if (key < 0) break;
        g.kill(key);
        r.deleted.push_back(key);
        r.bytes_freed += store[key].size_mb;
    }
    r.runtime_s = now_s() - t0;
    return r;
}

EvictionResult policy_lfu(const Store& store, MemoryGraph& g, double required_mb,
                          const PolicyParams&) {
    const double t0 = now_s();
    using Item = std::pair<int, int>;  // (access count, id)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (std::size_t i = 0; i < store.size(); ++i)
        heap.push({store[int(i)].access_count, int(i)});

    EvictionResult r;
    r.policy = "lfu";
    r.required_mb = required_mb;
    while (r.bytes_freed < required_mb && !heap.empty()) {
        const int v = heap.top().second;
        heap.pop();
        g.kill(v);
        r.deleted.push_back(v);
        r.bytes_freed += store[v].size_mb;
    }
    r.runtime_s = now_s() - t0;
    return r;
}

EvictionResult policy_size_greedy(const Store& store, MemoryGraph& g, double required_mb,
                                  const PolicyParams&) {
    const double t0 = now_s();
    return evict_in_order("size_greedy", store, g, required_mb,
                          order_by(store.size(), store.sizes(), true), t0);
}

// ------------------------------------------------------------------ CAFA
EvictionResult cafa(const Store& store, MemoryGraph& g, double required_mb,
                    const PolicyParams& p, const std::string& name) {
    const double t0 = now_s();
    const int n = static_cast<int>(store.size());

    std::vector<double> static_local;
    if (!p.static_part) static_local = static_features(store).combined;
    const std::vector<double>& static_part = p.static_part ? *p.static_part : static_local;

    const std::vector<double> sizes = store.sizes();
    const bool uses_graph = p.w_context > 0.0 || p.bridge_repair;
    const bool needs_tarjan = uses_graph && p.cut_weight > 0.0;

    std::unique_ptr<StructuralIndex> index;
    std::vector<double> wdeg(std::size_t(n), 0.0);
    std::vector<double> context(std::size_t(n), 0.0);
    double damage_scale = 1.0, max_wdeg = 1.0;

    if (uses_graph) {
        for (int v = 0; v < n; ++v)
            if (g.alive(v)) wdeg[std::size_t(v)] = g.weighted_degree(v);
        max_wdeg = *std::max_element(wdeg.begin(), wdeg.end());
        if (max_wdeg <= 0.0) max_wdeg = 1.0;

        if (needs_tarjan) {
            index = std::make_unique<StructuralIndex>(g);
            index->rebuild();
            // Calibrate against the worst cut actually present at t=0 and then
            // hold that scale fixed, so scores do not drift under us.
            damage_scale = std::max(1.0, *std::max_element(index->damage().begin(),
                                                           index->damage().end()));
            structural_score(index->damage(), wdeg, damage_scale, max_wdeg, p.cut_weight,
                             p.transform, context);
        } else {
            const std::vector<double> zero(std::size_t(n), 0.0);
            structural_score(zero, wdeg, damage_scale, max_wdeg, p.cut_weight, p.transform,
                             context);
        }
    }

    // Two-argument form on purpose: `std::vector<double> value(std::size_t(n))`
    // is the most vexing parse and declares a function, not a vector.
    std::vector<double> value(std::size_t(n), 0.0);
    std::vector<double> rho(std::size_t(n), 0.0);
    for (int v = 0; v < n; ++v) {
        value[std::size_t(v)] = retention_value(static_part[std::size_t(v)],
                                                context[std::size_t(v)], p.w_context);
        rho[std::size_t(v)] = value_density(value[std::size_t(v)], sizes[std::size_t(v)]);
    }

    std::vector<long long> version(std::size_t(n), 0);
    std::vector<double> value_lost(std::size_t(n), 0.0);
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;
    {
        std::vector<HeapEntry> seed;
        seed.reserve(std::size_t(n));
        for (int v = 0; v < n; ++v)
            if (g.alive(v)) seed.push_back({rho[std::size_t(v)], 0, v});
        heap = std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp>(
            HeapCmp{}, std::move(seed));
    }

    EvictionResult r;
    r.policy = name;
    r.required_mb = required_mb;

    while (r.bytes_freed < required_mb && !heap.empty()) {
        const HeapEntry e = heap.top();
        heap.pop();
        // Lazy deletion: std::priority_queue has no decrease-key, so stale
        // entries are pushed alongside and skipped here.
        if (!g.alive(e.node) || e.version != version[std::size_t(e.node)]) continue;

        g.kill(e.node);
        r.deleted.push_back(e.node);
        value_lost[std::size_t(e.node)] = value[std::size_t(e.node)];
        r.bytes_freed += sizes[std::size_t(e.node)];

        if (p.mode != Mode::Local || !uses_graph) continue;

        std::vector<int> neigh = g.neighbours(e.node);

        if (p.bridge_repair && neigh.size() > 1) {
            std::vector<int> ranked = neigh;
            std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
                double wa = 0.0, wb = 0.0;
                for (const auto& ed : g.adj(e.node)) {
                    if (ed.to == a) wa = ed.w;
                    if (ed.to == b) wb = ed.w;
                }
                if (wa != wb) return wa > wb;
                return a < b;
            });
            if (ranked.size() > kMaxBridgeDegree) ranked.resize(kMaxBridgeDegree);
            for (std::size_t a = 0; a < ranked.size(); ++a)
                for (std::size_t b = a + 1; b < ranked.size(); ++b) {
                    if (g.has_edge(ranked[a], ranked[b])) continue;
                    const double s = store.similarity(ranked[a], ranked[b]);
                    if (s >= p.bridge_threshold) {
                        g.add_edge(ranked[a], ranked[b], s);
                        ++r.bridges_added;
                    }
                }
        }

        // Only the component e.node was in can have changed, so re-run Tarjan
        // on that alone. With cut_weight 0 there is no structure to refresh
        // and only the immediate neighbours' degrees move.
        const std::vector<int>& changed =
            index ? index->update_after_delete(e.node) : neigh;
        if (changed.empty()) continue;

        for (int u : changed) {
            wdeg[std::size_t(u)] = g.weighted_degree(u);
            const double dmg = index ? index->damage()[std::size_t(u)] : 0.0;
            context[std::size_t(u)] = structural_score_one(dmg, wdeg[std::size_t(u)],
                                                           damage_scale, max_wdeg,
                                                           p.cut_weight, p.transform);
            value[std::size_t(u)] = retention_value(static_part[std::size_t(u)],
                                                    context[std::size_t(u)], p.w_context);
            rho[std::size_t(u)] = value_density(value[std::size_t(u)], sizes[std::size_t(u)]);
            ++version[std::size_t(u)];
            heap.push({rho[std::size_t(u)], version[std::size_t(u)], u});
        }
    }

    if (p.restore_overshoot)
        r.restored = repair_overshoot(store, g, r.deleted, value_lost, r.bytes_freed,
                                      required_mb);
    if (index) {
        r.analysis_calls = index->analysis_calls;
        r.nodes_visited = index->nodes_visited;
    }
    r.runtime_s = now_s() - t0;
    return r;
}

// -------------------------------------------------------------- variants
EvictionResult policy_value_greedy(const Store& store, MemoryGraph& g, double required_mb,
                                   const PolicyParams& p) {
    PolicyParams q = p;
    q.w_context = 0.0;
    q.mode = Mode::Static;
    q.bridge_repair = false;
    return cafa(store, g, required_mb, q, "value_greedy");
}

EvictionResult policy_degree_cafa(const Store& store, MemoryGraph& g, double required_mb,
                                  const PolicyParams& p) {
    PolicyParams q = p;
    q.cut_weight = 0.0;  // graph signal without Tarjan
    q.mode = Mode::Local;
    q.bridge_repair = false;
    return cafa(store, g, required_mb, q, "degree_cafa");
}

EvictionResult policy_cafa_static(const Store& store, MemoryGraph& g, double required_mb,
                                  const PolicyParams& p) {
    PolicyParams q = p;
    q.mode = Mode::Static;
    q.bridge_repair = false;
    return cafa(store, g, required_mb, q, "cafa_static");
}

EvictionResult policy_cafa_local(const Store& store, MemoryGraph& g, double required_mb,
                                 const PolicyParams& p) {
    PolicyParams q = p;
    q.mode = Mode::Local;
    q.bridge_repair = false;
    return cafa(store, g, required_mb, q, "cafa_local");
}

EvictionResult policy_cafa_repair(const Store& store, MemoryGraph& g, double required_mb,
                                  const PolicyParams& p) {
    PolicyParams q = p;
    q.mode = Mode::Local;
    q.bridge_repair = true;
    return cafa(store, g, required_mb, q, "cafa_repair");
}

const std::vector<std::pair<std::string, PolicyFn>>& policy_registry() {
    static const std::vector<std::pair<std::string, PolicyFn>> reg = {
        {"random", &policy_random},
        {"fifo", &policy_fifo},
        {"lru", &policy_lru},
        {"lfu", &policy_lfu},
        {"size_greedy", &policy_size_greedy},
        {"value_greedy", &policy_value_greedy},  // the ablation: CAFA, w_C = 0
        {"degree_cafa", &policy_degree_cafa},
        {"cafa_static", &policy_cafa_static},
        {"cafa_local", &policy_cafa_local},
        {"cafa_repair", &policy_cafa_repair},
    };
    return reg;
}

PolicyFn find_policy(const std::string& name) {
    for (const auto& [k, fn] : policy_registry())
        if (k == name) return fn;
    return nullptr;
}

}  // namespace memora
