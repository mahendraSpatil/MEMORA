// Benchmark harness: one CSV row per (policy, configuration, seed).
//
// Every chart, every table and any dashboard read this CSV. Metrics are never
// computed inside a UI, because that couples the results to the part of the
// project most likely to be unfinished the night before.
//
// Graph construction time is reported separately from eviction time. It is
// not part of the policy being compared, and folding it in would swamp the
// differences between policies.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace memora {

struct Config {
    int n = 1500;
    int n_categories = 12;
    double target_degree = 4.0;
    double bridge_fraction = 0.05;
    double budget_frac = 0.25;
    double importance_noise = 0.0;
    double importance_degree_rho = 0.0;
    double w_context = 0.30;
    double cut_weight = 0.70;
    std::string transform = "sqrt";
    int n_queries = 500;
    std::uint64_t seed = 0;
};

using Row = std::vector<std::pair<std::string, std::string>>;

// Build one dataset and run every named policy against the same byte budget.
std::vector<Row> run_config(const Config& cfg, const std::vector<std::string>& policies);

struct Sweep {
    std::string name;
    std::string description;
    std::vector<Row> (*run)(bool quiet);
};

const std::vector<Sweep>& sweeps();
const Sweep* find_sweep(const std::string& name);

std::vector<Row> knapsack_gap(bool quiet);

void write_csv(const std::vector<Row>& rows, const std::string& path);

}  // namespace memora
