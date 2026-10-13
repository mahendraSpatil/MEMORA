// The memory graph and its construction.
//
// Two things in here matter for the report.
//
// 1. Construction is the real bottleneck, not Tarjan. Comparing every pair of
//    memories against a similarity threshold is O(V^2): 5e7 comparisons at
//    V = 10,000, which costs more than every eviction run put together.
//    build_graph() blocks by category and samples a fixed number of
//    cross-category candidates, so the cost is
//    O(sum_c |cat_c|^2 + V * cross_samples).
//
// 2. Density is a parameter, not an accident. The builder keeps a mutual
//    nearest-neighbour backbone and then fills a global edge budget, so
//    target_degree is hit almost exactly. Average degree is the variable that
//    decides whether structural awareness pays off, so it has to be dialled
//    directly rather than chased through a similarity threshold.
//
// Deletion is a flag flip. Adjacency is never mutated on delete, which makes
// both deletion and restoration O(1) and the overshoot repair pass free.
#pragma once

#include <cstdint>
#include <vector>

#include "memora/memory.hpp"

namespace memora {

struct Edge {
    int to;
    double w;
};

class MemoryGraph {
public:
    explicit MemoryGraph(int n = 0) : n_(n), alive_(static_cast<std::size_t>(n), 1),
                                      adj_(static_cast<std::size_t>(n)) {}

    int n() const { return n_; }
    bool alive(int v) const { return alive_[static_cast<std::size_t>(v)] != 0; }
    void kill(int v) { alive_[static_cast<std::size_t>(v)] = 0; }
    void revive(int v) { alive_[static_cast<std::size_t>(v)] = 1; }
    const std::vector<std::uint8_t>& alive_mask() const { return alive_; }

    void add_edge(int u, int v, double w);
    bool has_edge(int u, int v) const;

    // Sort every adjacency list by neighbour id. Hash-container iteration
    // order decides the order edges are inserted, which decides DFS order,
    // which can decide which of two equally-scored memories is evicted.
    // Sorting once pins the run down so results are reproducible.
    void sort_adjacency();

    const std::vector<Edge>& adj(int v) const { return adj_[static_cast<std::size_t>(v)]; }

    int degree(int v) const;
    double weighted_degree(int v) const;
    std::vector<int> neighbours(int v) const;

    int live_count() const;
    long long edge_count() const;

    // Component id per node in the *undeleted* graph, computed once and
    // cached: it is the denominator of every retention metric, i.e. how many
    // relationships existed among the nodes a policy chose to keep.
    const std::vector<int>& original_components() const;

private:
    int n_;
    std::vector<std::uint8_t> alive_;
    std::vector<std::vector<Edge>> adj_;
    mutable std::vector<int> orig_comp_;
};

struct GraphParams {
    double target_degree = 4.0;
    int cross_samples = 32;
    double min_similarity = 0.15;
    int candidates_per_node = 0;  // 0 => derived from target_degree
    std::uint64_t seed = 0;
};

MemoryGraph build_graph(const Store& store, const GraphParams& p);

// Component ids and sizes over the *live* subgraph. O(V + E).
void connected_components(const MemoryGraph& g, std::vector<int>& comp_out,
                          std::vector<int>& sizes_out);

}  // namespace memora
