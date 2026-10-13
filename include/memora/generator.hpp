// Synthetic memory generator with a planted cluster structure.
//
// Why not "same category -> edge": that rule produces near-cliques with
// almost no cut vertices, and then every eviction policy scores identically.
// A generator that cannot produce bridges cannot show a structural method
// working *or* failing, so it cannot be used to test one.
//
// This plants:
//   * n_categories topic clusters, as centroids in a latent space;
//   * ordinary memories drawn near one centroid;
//   * bridge_fraction bridge memories drawn between *two* centroids.
//
// Bridges are the memories CAFA is supposed to protect. Their number is a
// sweep parameter precisely so the experiments can show how much of CAFA's
// advantage depends on them existing.
#pragma once

#include <cstdint>
#include <vector>

#include "memora/memory.hpp"

namespace memora {

struct GeneratorParams {
    int n = 1500;
    int n_categories = 12;
    double bridge_fraction = 0.05;
    int dim = 16;
    double cluster_spread = 0.55;
    double size_log_mu = 0.9;
    double size_log_sigma = 0.9;
    double max_age_days = 365.0;
    // Standard deviation of the gap between what a policy can see
    // (observed_importance) and what evaluation scores (true_importance).
    // 0.0 is the optimistic case; sweep upward to test robustness.
    double importance_noise = 0.0;
    std::uint64_t seed = 0;
};

Store generate_store(const GeneratorParams& p);

// Re-rank importance so it correlates with node degree at strength rho.
//
// If important memories are already the well-connected ones, LRU and LFU do
// accidentally well and CAFA has little left to add; at negative rho the
// opposite holds. Sweeping this is what separates "CAFA works" from "this
// dataset happened to be kind".
void apply_importance_degree_correlation(Store& store,
                                         const std::vector<double>& degrees,
                                         double rho, std::uint64_t seed);

}  // namespace memora
