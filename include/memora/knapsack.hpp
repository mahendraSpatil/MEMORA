// Exact reference for small instances.
//
// Freeing at least B bytes at minimum value lost is a minimum-cost covering
// knapsack: NP-hard, which is what justifies the greedy. But "greedy is
// reasonable because the problem is hard" is a much weaker claim than "greedy
// lands within 4% of optimal on 120-memory instances", and the second one
// costs an O(V*B) dynamic program to obtain.
//
// Run this on the small datasets only and report the gap.
#pragma once

#include <vector>

namespace memora {

// Minimum total value lost by any subset freeing at least required_mb.
// Sizes are discretised at `scale` units per MB.
double optimal_value_lost(const std::vector<double>& sizes_mb,
                          const std::vector<double>& values, double required_mb,
                          double scale = 1.0);

// Exhaustive reference for the test suite. Use with <= 20 items.
double brute_force_value_lost(const std::vector<double>& sizes_mb,
                              const std::vector<double>& values, double required_mb);

}  // namespace memora
