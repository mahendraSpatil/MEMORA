// Evaluation metrics.
//
// The original proposal's context measure -- 1 - fragmentation_after /
// max_possible_fragmentation -- has two problems. Maximum possible
// fragmentation is not well defined, and a component count is not
// proportional to damage: splitting off one leaf and halving the graph both
// add exactly one component.
//
// Everything here is therefore computed *relative to the survivors*. Take the
// set S a policy kept; the denominator is how many pairs of S were connected
// in the original graph, the numerator is how many still are. Deleting a
// memory that lay on no path between survivors then costs nothing, which is
// the correct behaviour and is not what a raw component count does.
//
// Three validity threats these metrics exist to close:
//
//  1. Importance is both an input and a score. Policies read
//     observed_importance; importance_retained scores true_importance. Sweep
//     the noise between them.
//  2. The objective and the metric are the same quantity. CAFA minimises pair
//     damage and pairwise_retention measures pair damage, so
//     context_query_success is reported beside it: a held-out sample of
//     related pairs drawn *before* any deletion, which CAFA never optimised.
//  3. Unequal work. overshoot and n_deleted appear on every row, so an unfair
//     comparison is visible rather than hidden.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "memora/graph.hpp"
#include "memora/memory.hpp"
#include "memora/policies.hpp"

namespace memora {

using QueryPair = std::pair<int, int>;

// Sample related pairs from the intact graph.
//
// Sampled once per dataset and shared by every policy, so all of them are
// graded on exactly the same questions. Pairs are drawn at distance 2-3:
// adjacent pairs are too easy, unrelated pairs measure nothing.
std::vector<QueryPair> sample_context_queries(const MemoryGraph& g, int n_queries,
                                              int max_depth, std::uint64_t seed);

struct Metrics {
    std::string policy;
    int n_deleted = 0;
    double bytes_freed_mb = 0.0;
    double required_mb = 0.0;
    double overshoot = 1.0;
    double runtime_s = 0.0;

    double pairwise_retention = 1.0;
    double neighbourhood_retention = 1.0;
    double lcc_ratio = 1.0;
    int components_after = 0;
    int components_before = 0;
    double importance_retained = 1.0;
    double bridge_retention = 1.0;
    int survivors = 0;

    double context_query_success = 1.0;
    double context_query_survival = 1.0;
    double query_coverage = 1.0;

    int restored = 0;
    long long bridges_added = 0;
    long long analysis_calls = 0;
    long long nodes_visited = 0;
};

Metrics evaluate(const Store& store, const MemoryGraph& original,
                 const MemoryGraph& after, const EvictionResult& result,
                 const std::vector<QueryPair>& queries);

}  // namespace memora
