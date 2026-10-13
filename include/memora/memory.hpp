// Memory records and the in-memory store.
//
// The split between true_importance and observed_importance is deliberate and
// load-bearing: eviction policies may only read the *observed* value, and
// evaluation scores the *true* one. Without that split, a policy that reads
// importance is graded on the very number it was handed, and the comparison
// proves nothing at all.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace memora {

struct Memory {
    int id = 0;
    int category = 0;
    double size_mb = 0.0;
    double true_importance = 0.0;      // ground truth, evaluation only
    double observed_importance = 0.0;  // noisy view, this is what policies see
    double age_days = 0.0;             // days since creation
    double recency_days = 0.0;         // days since last access
    int access_count = 0;
    bool is_bridge = false;
    std::string name;
};

// A fixed collection of memories indexed 0..n-1.
//
// Ids are contiguous ints on purpose: string keys and pointer-chasing
// adjacency cost several times more in the Tarjan inner loop at V = 100,000
// for no benefit. Display names live in the record, never in the hot path.
class Store {
public:
    std::vector<Memory> items;
    std::vector<double> vecs;  // flat, n * dim, each row L2-normalised
    int dim = 0;

    std::size_t size() const { return items.size(); }
    bool empty() const { return items.empty(); }
    const Memory& operator[](int i) const { return items[static_cast<std::size_t>(i)]; }

    double total_size_mb() const {
        double t = 0.0;
        for (const auto& m : items) t += m.size_mb;
        return t;
    }

    // Rows are unit vectors, so cosine similarity is a plain dot product.
    double similarity(int i, int j) const {
        const double* a = &vecs[static_cast<std::size_t>(i) * dim];
        const double* b = &vecs[static_cast<std::size_t>(j) * dim];
        double s = 0.0;
        for (int k = 0; k < dim; ++k) s += a[k] * b[k];
        return s;
    }

    // Column views the scoring code wants as plain arrays.
    std::vector<double> sizes() const;
    std::vector<double> observed_importance() const;
    std::vector<double> true_importance() const;
    std::vector<double> age_days() const;
    std::vector<double> recency_days() const;
    std::vector<double> access_counts() const;
};

}  // namespace memora
