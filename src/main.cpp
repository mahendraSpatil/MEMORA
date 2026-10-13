// Memora command line.
//
//   memora --demo                       one configuration, table on stdout
//   memora --list                       available sweeps
//   memora --sweep density --out f.csv  one sweep to CSV
//   memora --all                        every sweep + the knapsack gap
//   memora --knapsack                   greedy vs exact DP on small instances
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "memora/benchmark.hpp"
#include "memora/generator.hpp"
#include "memora/graph.hpp"
#include "memora/metrics.hpp"
#include "memora/policies.hpp"
#include "memora/scoring.hpp"

using namespace memora;

namespace {

void print_usage() {
    std::cout <<
        "Memora - context-aware memory eviction\n\n"
        "  --demo                 run one configuration and print a table\n"
        "  --list                 list available sweeps\n"
        "  --sweep <name>         run one sweep\n"
        "  --all                  run every sweep and the knapsack gap\n"
        "  --knapsack             greedy vs exact DP on small instances\n"
        "  --out <path>           CSV output path (default results/<sweep>.csv)\n"
        "  --n <int>              memories            (demo, default 1500)\n"
        "  --degree <float>       average degree      (demo, default 4)\n"
        "  --budget <float>       fraction to free    (demo, default 0.25)\n"
        "  --w-context <float>    context weight      (demo, default 0.3)\n"
        "  --cut-weight <float>   damage vs degree    (demo, default 0.7)\n"
        "  --seed <int>           random seed         (demo, default 0)\n"
        "  --quiet                suppress progress\n";
}

void run_demo(const Config& cfg) {
    std::cout << "n=" << cfg.n << "  degree=" << cfg.target_degree
              << "  budget=" << cfg.budget_frac * 100 << "%"
              << "  w_context=" << cfg.w_context << "  cut_weight=" << cfg.cut_weight
              << "  seed=" << cfg.seed << "\n\n";

    std::vector<std::string> names;
    for (const auto& [k, fn] : policy_registry()) {
        (void)fn;
        names.push_back(k);
    }
    const std::vector<Row> rows = run_config(cfg, names);

    const std::vector<std::string> cols = {
        "policy", "n_deleted", "overshoot", "pairwise_retention",
        "neighbourhood_retention", "context_query_success", "importance_retained",
        "bridge_retention", "runtime_s"};
    const std::vector<std::string> head = {
        "policy", "del", "over", "pair_ret", "nbr_ret", "query_ok", "imp_ret",
        "bridge", "time_s"};

    std::vector<std::size_t> w(cols.size());
    for (std::size_t i = 0; i < cols.size(); ++i) w[i] = head[i].size();
    for (const auto& row : rows)
        for (std::size_t i = 0; i < cols.size(); ++i)
            for (const auto& [k, v] : row)
                if (k == cols[i]) w[i] = std::max(w[i], v.size());

    for (std::size_t i = 0; i < cols.size(); ++i)
        std::cout << std::setw(int(w[i]) + 2) << head[i];
    std::cout << "\n";
    for (std::size_t i = 0; i < cols.size(); ++i)
        std::cout << std::string(w[i] + 2, '-');
    std::cout << "\n";
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < cols.size(); ++i) {
            std::string val;
            for (const auto& [k, v] : row)
                if (k == cols[i]) val = v;
            std::cout << std::setw(int(w[i]) + 2) << val;
        }
        std::cout << "\n";
    }
    std::cout << "\nvalue_greedy is CAFA with w_C = 0: the gap between that row and\n"
                 "cafa_local is what the graph is actually buying.\n";
}

double darg(int argc, char** argv, int i) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return std::stod(argv[i + 1]);
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    std::string sweep_name, out_path;
    bool do_demo = false, do_list = false, do_all = false, do_knap = false, quiet = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--demo") do_demo = true;
            else if (a == "--list") do_list = true;
            else if (a == "--all") do_all = true;
            else if (a == "--knapsack") do_knap = true;
            else if (a == "--quiet") quiet = true;
            else if (a == "--help" || a == "-h") { print_usage(); return 0; }
            else if (a == "--sweep") { sweep_name = argv[++i]; }
            else if (a == "--out") { out_path = argv[++i]; }
            else if (a == "--n") { cfg.n = int(darg(argc, argv, i)); ++i; }
            else if (a == "--degree") { cfg.target_degree = darg(argc, argv, i); ++i; }
            else if (a == "--budget") { cfg.budget_frac = darg(argc, argv, i); ++i; }
            else if (a == "--w-context") { cfg.w_context = darg(argc, argv, i); ++i; }
            else if (a == "--cut-weight") { cfg.cut_weight = darg(argc, argv, i); ++i; }
            else if (a == "--seed") { cfg.seed = std::uint64_t(darg(argc, argv, i)); ++i; }
            else { std::cerr << "unknown argument: " << a << "\n"; print_usage(); return 2; }
        }
    } catch (const std::exception& e) {
        std::cerr << "bad arguments: " << e.what() << "\n";
        return 2;
    }

    if (do_list) {
        std::cout << "sweeps:\n";
        for (const auto& s : sweeps())
            std::cout << "  " << std::left << std::setw(14) << s.name << s.description << "\n";
        return 0;
    }
    if (do_knap) {
        write_csv(knapsack_gap(quiet), out_path.empty() ? "results/knapsack.csv" : out_path);
        return 0;
    }
    if (do_all) {
        for (const auto& s : sweeps()) {
            std::cout << "== sweep: " << s.name << " (" << s.description << ")\n";
            write_csv(s.run(quiet), "results/" + s.name + ".csv");
        }
        std::cout << "== knapsack gap\n";
        write_csv(knapsack_gap(quiet), "results/knapsack.csv");
        return 0;
    }
    if (!sweep_name.empty()) {
        const Sweep* s = find_sweep(sweep_name);
        if (!s) {
            std::cerr << "unknown sweep: " << sweep_name << " (try --list)\n";
            return 2;
        }
        write_csv(s->run(quiet), out_path.empty() ? "results/" + s->name + ".csv" : out_path);
        return 0;
    }
    if (do_demo || argc == 1) {
        run_demo(cfg);
        return 0;
    }
    print_usage();
    return 0;
}
