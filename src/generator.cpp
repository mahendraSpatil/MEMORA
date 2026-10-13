#include "memora/generator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>

namespace memora {

namespace {

const char* const kTopics[] = {
    "lecture notes", "assignment", "exam paper", "lab report", "meeting notes",
    "screenshot", "invoice", "photo", "chat log", "bookmark", "email thread",
    "code snippet", "reference paper", "todo list", "contact card",
};
constexpr int kNumTopics = int(sizeof(kTopics) / sizeof(kTopics[0]));

double sample_beta(std::mt19937_64& rng, double a, double b) {
    std::gamma_distribution<double> ga(a, 1.0), gb(b, 1.0);
    const double x = ga(rng), y = gb(rng);
    const double s = x + y;
    return s > 0.0 ? x / s : 0.5;
}

std::vector<double> rank_uniform_signed(const std::vector<double>& a) {
    const std::size_t n = a.size();
    std::vector<double> out(n, 0.0);
    if (n < 2) return out;
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&a](std::size_t x, std::size_t y) { return a[x] < a[y]; });
    for (std::size_t r = 0; r < n; ++r)
        out[order[r]] = (double(r) / double(n - 1)) * 2.0 - 1.0;
    return out;
}

}  // namespace

Store generate_store(const GeneratorParams& p) {
    std::mt19937_64 rng(p.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    Store store;
    store.dim = p.dim;
    store.items.reserve(static_cast<std::size_t>(p.n));
    store.vecs.assign(static_cast<std::size_t>(p.n) * p.dim, 0.0);

    // Category centroids, unit length.
    std::vector<double> centroids(static_cast<std::size_t>(p.n_categories) * p.dim);
    for (int c = 0; c < p.n_categories; ++c) {
        double norm = 0.0;
        for (int k = 0; k < p.dim; ++k) {
            const double x = gauss(rng);
            centroids[static_cast<std::size_t>(c) * p.dim + k] = x;
            norm += x * x;
        }
        norm = std::sqrt(norm);
        if (norm <= 0.0) norm = 1.0;
        for (int k = 0; k < p.dim; ++k) centroids[static_cast<std::size_t>(c) * p.dim + k] /= norm;
    }

    // Pick the bridge memories up front: they span two centroids and are
    // exactly the items a structural policy is supposed to protect.
    const int n_bridges = int(std::lround(p.n * p.bridge_fraction));
    std::unordered_set<int> bridges;
    if (n_bridges > 0) {
        std::vector<int> ids(static_cast<std::size_t>(p.n));
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        bridges.insert(ids.begin(), ids.begin() + std::min(n_bridges, p.n));
    }

    std::uniform_int_distribution<int> pick_cat(0, std::max(0, p.n_categories - 1));
    std::lognormal_distribution<double> lognorm(p.size_log_mu, p.size_log_sigma);

    for (int i = 0; i < p.n; ++i) {
        Memory m;
        m.id = i;
        m.category = pick_cat(rng);
        m.is_bridge = bridges.count(i) > 0;

        double* v = &store.vecs[static_cast<std::size_t>(i) * p.dim];
        const double* base = &centroids[static_cast<std::size_t>(m.category) * p.dim];
        if (m.is_bridge && p.n_categories > 1) {
            std::uniform_int_distribution<int> pick_other(0, p.n_categories - 2);
            int other = pick_other(rng);
            if (other >= m.category) ++other;
            const double* ob = &centroids[static_cast<std::size_t>(other) * p.dim];
            const double mix = 0.35 + 0.30 * uni(rng);
            for (int k = 0; k < p.dim; ++k)
                v[k] = mix * base[k] + (1.0 - mix) * ob[k] + p.cluster_spread * gauss(rng);
        } else {
            for (int k = 0; k < p.dim; ++k) v[k] = base[k] + p.cluster_spread * gauss(rng);
        }
        double norm = 0.0;
        for (int k = 0; k < p.dim; ++k) norm += v[k] * v[k];
        norm = std::sqrt(norm);
        if (norm <= 0.0) norm = 1.0;
        for (int k = 0; k < p.dim; ++k) v[k] /= norm;

        m.size_mb = std::clamp(lognorm(rng), 0.1, 400.0);
        m.true_importance = std::clamp(sample_beta(rng, 2.0, 3.0), 0.0, 1.0);
        if (p.importance_noise > 0.0) {
            std::normal_distribution<double> noise(0.0, p.importance_noise);
            m.observed_importance = std::clamp(m.true_importance + noise(rng), 0.0, 1.0);
        } else {
            m.observed_importance = m.true_importance;
        }
        m.age_days = uni(rng) * p.max_age_days;
        // Recent access is more likely on young, important memories.
        m.recency_days = m.age_days * sample_beta(rng, 1.0, 1.0 + 3.0 * m.true_importance);
        const double lam =
            1.0 + 25.0 * m.true_importance * std::exp(-m.recency_days / 60.0);
        std::poisson_distribution<int> pois(lam);
        m.access_count = pois(rng);
        m.name = std::string(kTopics[m.category % kNumTopics]) + " #" + std::to_string(i);

        store.items.push_back(std::move(m));
    }
    return store;
}

void apply_importance_degree_correlation(Store& store, const std::vector<double>& degrees,
                                         double rho, std::uint64_t seed) {
    if (std::abs(rho) < 1e-9) return;
    const std::size_t n = store.size();
    std::mt19937_64 rng(seed ^ 0xD1B54A32D192ED03ULL);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> noise(n);
    for (auto& x : noise) x = gauss(rng);

    const std::vector<double> deg_rank = rank_uniform_signed(degrees);
    const std::vector<double> noise_rank = rank_uniform_signed(noise);

    std::vector<double> blended(n);
    const double w = std::sqrt(std::max(0.0, 1.0 - rho * rho));
    for (std::size_t i = 0; i < n; ++i) blended[i] = rho * deg_rank[i] + w * noise_rank[i];

    // Re-attach the same multiset of importances, ordered to match `blended`.
    std::vector<double> sorted_importance = store.true_importance();
    std::sort(sorted_importance.begin(), sorted_importance.end());

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&blended](std::size_t x, std::size_t y) { return blended[x] < blended[y]; });

    for (std::size_t r = 0; r < n; ++r) {
        Memory& m = store.items[order[r]];
        const double delta = m.observed_importance - m.true_importance;
        m.true_importance = sorted_importance[r];
        m.observed_importance = std::clamp(m.true_importance + delta, 0.0, 1.0);
    }
}

}  // namespace memora
