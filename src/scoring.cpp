#include "memora/scoring.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace memora {

std::vector<double> rank_normalize(const std::vector<double>& a) {
    const std::size_t n = a.size();
    std::vector<double> ranks(n, 0.0);
    if (n < 2) return ranks;

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&a](std::size_t x, std::size_t y) { return a[x] < a[y]; });

    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j + 1 < n && a[order[j + 1]] == a[order[i]]) ++j;
        const double avg = 0.5 * double(i + j);
        for (std::size_t k = i; k <= j; ++k) ranks[order[k]] = avg;
        i = j + 1;
    }
    const double denom = double(n - 1);
    for (auto& r : ranks) r /= denom;
    return ranks;
}

StaticFeatures static_features(const Store& store) {
    // Relative split of the non-structural terms, renormalised to sum to 1 so
    // that sweeping w_C never changes the overall scale of the value.
    constexpr double kImp = 0.35, kUse = 0.20, kFresh = 0.15;
    constexpr double kSum = kImp + kUse + kFresh;

    StaticFeatures f;
    f.importance = rank_normalize(store.observed_importance());

    const std::vector<double> recency_rank = rank_normalize(store.recency_days());
    std::vector<double> log_access = store.access_counts();
    for (auto& v : log_access) v = std::log1p(v);
    const std::vector<double> freq_rank = rank_normalize(log_access);

    const std::vector<double> age_rank = rank_normalize(store.age_days());

    const std::size_t n = store.size();
    f.usage.resize(n);
    f.freshness.resize(n);
    f.combined.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        f.usage[i] = 0.5 * (1.0 - recency_rank[i]) + 0.5 * freq_rank[i];
        f.freshness[i] = 1.0 - age_rank[i];
        f.combined[i] = (kImp * f.importance[i] + kUse * f.usage[i] +
                         kFresh * f.freshness[i]) / kSum;
    }
    return f;
}

Transform parse_transform(const std::string& s) {
    if (s == "sqrt") return Transform::Sqrt;
    if (s == "linear") return Transform::Linear;
    if (s == "log") return Transform::Log;
    throw std::invalid_argument("unknown transform: " + s);
}

double structural_score_one(double damage, double weighted_degree,
                            double damage_scale, double max_weighted_degree,
                            double cut_weight, Transform transform) {
    double d = damage / (damage_scale > 1.0 ? damage_scale : 1.0);
    if (d < 0.0) d = 0.0;
    if (d > 1.0) d = 1.0;
    switch (transform) {
        case Transform::Sqrt: d = std::sqrt(d); break;
        case Transform::Linear: break;
        case Transform::Log: d = std::log1p(d * 9.0) / std::log(10.0); break;
    }
    double loc = 0.0;
    if (max_weighted_degree > 0.0) {
        loc = weighted_degree / max_weighted_degree;
        if (loc > 1.0) loc = 1.0;
    }
    return cut_weight * d + (1.0 - cut_weight) * loc;
}

void structural_score(const std::vector<double>& damage,
                      const std::vector<double>& weighted_degree,
                      double damage_scale, double max_weighted_degree,
                      double cut_weight, Transform transform,
                      std::vector<double>& out) {
    out.resize(damage.size());
    for (std::size_t i = 0; i < damage.size(); ++i)
        out[i] = structural_score_one(damage[i], weighted_degree[i], damage_scale,
                                      max_weighted_degree, cut_weight, transform);
}

}  // namespace memora
