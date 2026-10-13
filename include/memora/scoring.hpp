// Retention value, normalisation, and the value-density eviction key.
//
// The original proposal scored
//
//     F(m) = w1*A + w2*S + w3(1-I) + w4(1-R) + w5(1-C)
//
// which adds size to the value terms, i.e. treats "this file is big" as
// evidence the memory is worthless. It is not: size is the *benefit* of
// deleting, the other terms are the *cost*. Mixing them into one weighted sum
// leaves the weights uninterpretable and undefendable.
//
// Stated properly the task is: free at least B bytes while losing as little
// value as possible. That is a minimum-cost covering knapsack, which is
// NP-hard, so a greedy is the right tool -- and the right greedy key is value
// per byte:
//
//     V(m)   = (1 - w_C) * [utility terms] + w_C * C(m)
//     rho(m) = V(m) / size(m)        -> evict in increasing rho
//
// Size needs no weight of its own, which drops five tunable numbers to four,
// and w_C = 0 gives the value-greedy ablation for free: the identical policy
// with the graph switched off.
//
// Everything is rank-normalised rather than min-max normalised. Min-max lets
// one 2 GB outlier crush every other size to ~0; percentile rank costs
// O(V log V) and is robust.
#pragma once

#include <string>
#include <vector>

#include "memora/memory.hpp"

namespace memora {

// Percentile rank in [0, 1] with ties averaged.
std::vector<double> rank_normalize(const std::vector<double>& a);

struct StaticFeatures {
    std::vector<double> importance;
    std::vector<double> usage;
    std::vector<double> freshness;
    std::vector<double> combined;  // the part of the value that never moves
};

// Rank-normalise the metadata a policy is allowed to see.
// Reads observed_importance, never true_importance.
StaticFeatures static_features(const Store& store);

enum class Transform { Sqrt, Linear, Log };

Transform parse_transform(const std::string& s);

// Turn raw pair damage into a context score C(v) in [0, 1].
//
// damage_scale is the largest damage present in the *initial* graph.
// Normalising against total pairs instead looks more principled but crushes
// almost every node to ~0: in a 1,000-node graph, severing a five-node
// cluster destroys 1% of the pairs, so the term stops influencing anything.
// Calibrating against the worst case actually present keeps the full [0, 1]
// range available, and fixing the scale at t = 0 keeps it stable while the
// graph shrinks.
//
// cut_weight splits C between disconnection damage and the local context a
// node carries in its own edges. Setting it to 0 gives the degree-only
// baseline -- the cheap graph signal CAFA has to beat to justify Tarjan.
//
// The transform is stateless on purpose: rank-normalising inside the eviction
// loop would cost O(V log V) per deletion. Sqrt spreads the lower end, where
// almost every node sits, without needing global state.
double structural_score_one(double damage, double weighted_degree,
                            double damage_scale, double max_weighted_degree,
                            double cut_weight, Transform transform);

void structural_score(const std::vector<double>& damage,
                      const std::vector<double>& weighted_degree,
                      double damage_scale, double max_weighted_degree,
                      double cut_weight, Transform transform,
                      std::vector<double>& out);

inline double retention_value(double static_part, double context, double w_context) {
    return (1.0 - w_context) * static_part + w_context * context;
}

inline double value_density(double value, double size_mb) {
    return value / (size_mb > 1e-9 ? size_mb : 1e-9);
}

}  // namespace memora
