#include "memora/knapsack.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace memora {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

double optimal_value_lost(const std::vector<double>& sizes_mb,
                          const std::vector<double>& values, double required_mb,
                          double scale) {
    const std::size_t n = sizes_mb.size();
    const long long budget = static_cast<long long>(std::ceil(required_mb * scale));
    if (budget <= 0) return 0.0;

    std::vector<long long> units(n);
    long long total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        units[i] = std::max<long long>(1, std::llround(sizes_mb[i] * scale));
        total += units[i];
    }
    if (total < budget) {
        double all = 0.0;
        for (double v : values) all += v;
        return all;
    }

    // dp[j] = minimum value lost by a subset that frees exactly j units,
    // with j saturating at the budget (freeing more than needed is allowed).
    std::vector<double> dp(static_cast<std::size_t>(budget) + 1, kInf);
    dp[0] = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const long long u = units[i];
        const double val = values[i];
        for (long long j = budget; j >= 0; --j) {
            const double cur = dp[static_cast<std::size_t>(j)];
            if (cur == kInf) continue;
            const long long k = (j + u > budget) ? budget : j + u;
            const double cand = cur + val;
            if (cand < dp[static_cast<std::size_t>(k)]) dp[static_cast<std::size_t>(k)] = cand;
        }
    }
    return dp[static_cast<std::size_t>(budget)];
}

double brute_force_value_lost(const std::vector<double>& sizes_mb,
                              const std::vector<double>& values, double required_mb) {
    const std::size_t n = sizes_mb.size();
    if (n > 20) throw std::invalid_argument("brute force is for tiny instances only");
    double best = kInf;
    const unsigned long long limit = 1ULL << n;
    for (unsigned long long mask = 0; mask < limit; ++mask) {
        double freed = 0.0, lost = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (mask & (1ULL << i)) {
                freed += sizes_mb[i];
                lost += values[i];
            }
        }
        if (freed >= required_mb && lost < best) best = lost;
    }
    return best;
}

}  // namespace memora
