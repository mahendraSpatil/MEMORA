#include "memora/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "memora/generator.hpp"
#include "memora/graph.hpp"
#include "memora/knapsack.hpp"
#include "memora/metrics.hpp"
#include "memora/policies.hpp"
#include "memora/scoring.hpp"
#include "memora/tarjan.hpp"

namespace memora {

namespace {

std::string fmt(double v, int prec = 6) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(prec);
    os << v;
    std::string s = os.str();
    if (s.find('.') != std::string::npos) {
        while (s.size() > 1 && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

std::string fmt(long long v) { return std::to_string(v); }
std::string fmt(int v) { return std::to_string(v); }

std::vector<std::string> all_policies() {
    std::vector<std::string> names;
    for (const auto& [k, fn] : policy_registry()) {
        (void)fn;
        names.push_back(k);
    }
    return names;
}

void add_config_columns(Row& row, const Config& c) {
    row.emplace_back("n", fmt(c.n));
    row.emplace_back("n_categories", fmt(c.n_categories));
    row.emplace_back("target_degree", fmt(c.target_degree, 3));
    row.emplace_back("bridge_fraction", fmt(c.bridge_fraction, 4));
    row.emplace_back("budget_frac", fmt(c.budget_frac, 4));
    row.emplace_back("importance_noise", fmt(c.importance_noise, 4));
    row.emplace_back("importance_degree_rho", fmt(c.importance_degree_rho, 4));
    row.emplace_back("w_context", fmt(c.w_context, 4));
    row.emplace_back("cut_weight", fmt(c.cut_weight, 4));
    row.emplace_back("transform", c.transform);
    row.emplace_back("seed", fmt(static_cast<long long>(c.seed)));
}

}  // namespace

std::vector<Row> run_config(const Config& cfg, const std::vector<std::string>& policies) {
    GeneratorParams gp;
    gp.n = cfg.n;
    gp.n_categories = cfg.n_categories;
    gp.bridge_fraction = cfg.bridge_fraction;
    gp.importance_noise = cfg.importance_noise;
    gp.seed = cfg.seed;
    Store store = generate_store(gp);

    GraphParams ggp;
    ggp.target_degree = cfg.target_degree;
    ggp.seed = cfg.seed;

    const double build_t0 =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    MemoryGraph graph = build_graph(store, ggp);
    const double build_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
            .count() -
        build_t0;

    if (std::abs(cfg.importance_degree_rho) > 1e-9) {
        std::vector<double> degrees(static_cast<std::size_t>(graph.n()));
        for (int v = 0; v < graph.n(); ++v) degrees[static_cast<std::size_t>(v)] = graph.degree(v);
        apply_importance_degree_correlation(store, degrees, cfg.importance_degree_rho, cfg.seed);
    }

    const std::vector<QueryPair> queries =
        sample_context_queries(graph, cfg.n_queries, 3, cfg.seed);
    const std::vector<double> static_part = static_features(store).combined;
    const double required = cfg.budget_frac * store.total_size_mb();
    const long long edges = graph.edge_count();

    std::vector<Row> rows;
    for (const std::string& name : policies) {
        PolicyFn fn = find_policy(name);
        if (!fn) {
            std::cerr << "unknown policy: " << name << "\n";
            continue;
        }
        PolicyParams pp;
        pp.w_context = cfg.w_context;
        pp.cut_weight = cfg.cut_weight;
        pp.transform = parse_transform(cfg.transform);
        pp.seed = cfg.seed;
        pp.static_part = &static_part;

        MemoryGraph working = graph;  // every policy starts from the same graph
        const EvictionResult res = fn(store, working, required, pp);
        const Metrics m = evaluate(store, graph, working, res, queries);

        Row row;
        row.emplace_back("policy", m.policy);
        row.emplace_back("n_deleted", fmt(m.n_deleted));
        row.emplace_back("survivors", fmt(m.survivors));
        row.emplace_back("bytes_freed_mb", fmt(m.bytes_freed_mb, 3));
        row.emplace_back("required_mb", fmt(m.required_mb, 3));
        row.emplace_back("overshoot", fmt(m.overshoot, 5));
        row.emplace_back("runtime_s", fmt(m.runtime_s, 6));
        row.emplace_back("pairwise_retention", fmt(m.pairwise_retention, 5));
        row.emplace_back("neighbourhood_retention", fmt(m.neighbourhood_retention, 5));
        row.emplace_back("lcc_ratio", fmt(m.lcc_ratio, 5));
        row.emplace_back("context_query_success", fmt(m.context_query_success, 5));
        row.emplace_back("context_query_survival", fmt(m.context_query_survival, 5));
        row.emplace_back("query_coverage", fmt(m.query_coverage, 5));
        row.emplace_back("importance_retained", fmt(m.importance_retained, 5));
        row.emplace_back("bridge_retention", fmt(m.bridge_retention, 5));
        row.emplace_back("components_before", fmt(m.components_before));
        row.emplace_back("components_after", fmt(m.components_after));
        row.emplace_back("restored", fmt(m.restored));
        row.emplace_back("bridges_added", fmt(m.bridges_added));
        row.emplace_back("analysis_calls", fmt(m.analysis_calls));
        row.emplace_back("nodes_visited", fmt(m.nodes_visited));
        add_config_columns(row, cfg);
        row.emplace_back("graph_build_s", fmt(build_s, 6));
        row.emplace_back("edges", fmt(edges));
        row.emplace_back("actual_degree", fmt(2.0 * double(edges) / std::max(1, graph.n()), 3));
        rows.push_back(std::move(row));
    }
    return rows;
}

// ------------------------------------------------------------------ sweeps
namespace {

void progress(bool quiet, const std::string& what, int i, int total) {
    if (quiet) return;
    std::cout << "  [" << i << "/" << total << "] " << what << std::endl;
}

std::vector<Row> sweep_main(bool quiet) {
    std::vector<Row> rows;
    for (int s = 0; s < 10; ++s) {
        Config c;
        c.seed = static_cast<std::uint64_t>(s);
        auto r = run_config(c, all_policies());
        rows.insert(rows.end(), r.begin(), r.end());
        progress(quiet, "seed=" + std::to_string(s), s + 1, 10);
    }
    return rows;
}

std::vector<Row> sweep_density(bool quiet) {
    const std::vector<double> degrees = {2, 3, 4, 6, 8, 12};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(degrees.size()) * 10;
    for (double d : degrees)
        for (int s = 0; s < 10; ++s) {
            Config c;
            c.target_degree = d;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, all_policies());
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "degree=" + fmt(d, 1) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

std::vector<Row> sweep_budget(bool quiet) {
    const std::vector<double> budgets = {0.05, 0.10, 0.25, 0.40, 0.50};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(budgets.size()) * 10;
    for (double b : budgets)
        for (int s = 0; s < 10; ++s) {
            Config c;
            c.budget_frac = b;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, all_policies());
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "budget=" + fmt(b, 2) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

std::vector<Row> sweep_weight(bool quiet) {
    const std::vector<double> ws = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    const std::vector<std::string> pol = {"cafa_local", "cafa_repair"};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(ws.size()) * 6;
    for (double w : ws)
        for (int s = 0; s < 6; ++s) {
            Config c;
            c.w_context = w;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, pol);
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "w_context=" + fmt(w, 2) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

std::vector<Row> sweep_cutweight(bool quiet) {
    const std::vector<double> cws = {0.0, 0.25, 0.5, 0.7, 0.9, 1.0};
    const std::vector<double> degs = {2, 4, 8};
    const std::vector<std::string> pol = {"cafa_local"};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(cws.size() * degs.size()) * 5;
    for (double cw : cws)
        for (double d : degs)
            for (int s = 0; s < 5; ++s) {
                Config c;
                c.cut_weight = cw;
                c.target_degree = d;
                c.seed = static_cast<std::uint64_t>(s);
                auto r = run_config(c, pol);
                rows.insert(rows.end(), r.begin(), r.end());
                progress(quiet, "cut_weight=" + fmt(cw, 2) + " degree=" + fmt(d, 1), ++i, total);
            }
    return rows;
}

std::vector<Row> sweep_noise(bool quiet) {
    const std::vector<double> noises = {0.0, 0.1, 0.2, 0.3};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(noises.size()) * 6;
    for (double nz : noises)
        for (int s = 0; s < 6; ++s) {
            Config c;
            c.importance_noise = nz;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, all_policies());
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "noise=" + fmt(nz, 2) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

std::vector<Row> sweep_correlation(bool quiet) {
    const std::vector<double> rhos = {-0.5, 0.0, 0.5};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(rhos.size()) * 6;
    for (double rho : rhos)
        for (int s = 0; s < 6; ++s) {
            Config c;
            c.importance_degree_rho = rho;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, all_policies());
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "rho=" + fmt(rho, 2) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

std::vector<Row> sweep_bridges(bool quiet) {
    const std::vector<double> fracs = {0.0, 0.01, 0.05, 0.10};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(fracs.size()) * 6;
    for (double f : fracs)
        for (int s = 0; s < 6; ++s) {
            Config c;
            c.bridge_fraction = f;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, all_policies());
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "bridges=" + fmt(f, 2) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

std::vector<Row> sweep_scale(bool quiet) {
    const std::vector<int> sizes = {100, 500, 1000, 2000, 5000, 10000};
    std::vector<Row> rows;
    int i = 0;
    const int total = int(sizes.size()) * 3;
    for (int n : sizes)
        for (int s = 0; s < 3; ++s) {
            Config c;
            c.n = n;
            c.n_queries = 200;
            c.seed = static_cast<std::uint64_t>(s);
            auto r = run_config(c, all_policies());
            rows.insert(rows.end(), r.begin(), r.end());
            progress(quiet, "n=" + std::to_string(n) + " seed=" + std::to_string(s), ++i, total);
        }
    return rows;
}

}  // namespace

std::vector<Row> knapsack_gap(bool quiet) {
    std::vector<Row> rows;
    const int kN = 120;
    for (int seed = 0; seed < 8; ++seed) {
        Config c;
        c.n = kN;
        c.n_categories = 6;
        c.n_queries = 100;
        c.seed = static_cast<std::uint64_t>(seed);

        GeneratorParams gp;
        gp.n = c.n;
        gp.n_categories = c.n_categories;
        gp.seed = c.seed;
        Store store = generate_store(gp);

        GraphParams ggp;
        ggp.target_degree = c.target_degree;
        ggp.seed = c.seed;
        MemoryGraph graph = build_graph(store, ggp);

        const std::vector<double> static_part = static_features(store).combined;
        const std::vector<double> sizes = store.sizes();
        const double required = c.budget_frac * store.total_size_mb();

        for (const std::string& name : {std::string("value_greedy"), std::string("cafa_local")}) {
            PolicyParams pp;
            pp.w_context = (name == "value_greedy") ? 0.0 : c.w_context;
            pp.cut_weight = c.cut_weight;
            pp.static_part = &static_part;
            pp.seed = c.seed;

            MemoryGraph working = graph;
            const EvictionResult res = find_policy(name)(store, working, required, pp);

            // Reconstruct the objective the policy actually optimised, so the
            // exact optimum is of the same function and the gap is meaningful.
            std::vector<double> value(store.size());
            if (pp.w_context == 0.0) {
                value = static_part;
            } else {
                MemoryGraph fresh = graph;
                StructuralIndex idx(fresh);
                idx.rebuild();
                std::vector<double> wdeg(store.size());
                for (int v = 0; v < fresh.n(); ++v)
                    wdeg[static_cast<std::size_t>(v)] = fresh.weighted_degree(v);
                const double dmax = std::max(1.0, *std::max_element(idx.damage().begin(),
                                                                    idx.damage().end()));
                const double wmax = std::max(1e-9, *std::max_element(wdeg.begin(), wdeg.end()));
                std::vector<double> ctx;
                structural_score(idx.damage(), wdeg, dmax, wmax, c.cut_weight,
                                 Transform::Sqrt, ctx);
                for (std::size_t v = 0; v < store.size(); ++v)
                    value[v] = retention_value(static_part[v], ctx[v], pp.w_context);
            }

            double greedy_lost = 0.0;
            for (int v : res.deleted) greedy_lost += value[static_cast<std::size_t>(v)];
            const double opt = optimal_value_lost(sizes, value, required, 1.0);

            Row row;
            row.emplace_back("policy", name);
            row.emplace_back("n", fmt(c.n));
            row.emplace_back("seed", fmt(static_cast<long long>(c.seed)));
            row.emplace_back("greedy_value_lost", fmt(greedy_lost, 5));
            row.emplace_back("optimal_value_lost", fmt(opt, 5));
            row.emplace_back("gap_pct",
                             fmt(100.0 * (greedy_lost - opt) / std::max(opt, 1e-9), 3));
            row.emplace_back("n_deleted", fmt(res.n_deleted()));
            row.emplace_back("overshoot", fmt(res.overshoot(), 5));
            rows.push_back(std::move(row));
        }
        progress(quiet, "knapsack seed=" + std::to_string(seed), seed + 1, 8);
    }
    return rows;
}

const std::vector<Sweep>& sweeps() {
    static const std::vector<Sweep> s = {
        {"main", "all policies, 10 seeds, default configuration", &sweep_main},
        {"density", "average degree 2..12 -- the key sweep", &sweep_density},
        {"budget", "eviction budget 5%..50% of total bytes", &sweep_budget},
        {"weight", "context weight 0..0.5 (justifies the weights)", &sweep_weight},
        {"cutweight", "Tarjan damage vs degree, at three densities", &sweep_cutweight},
        {"noise", "metadata noise: does CAFA degrade gracefully?", &sweep_noise},
        {"correlation", "importance vs degree correlation", &sweep_correlation},
        {"bridges", "how much depends on bridge memories existing", &sweep_bridges},
        {"scale", "V = 100..10000, for the timing plot", &sweep_scale},
    };
    return s;
}

const Sweep* find_sweep(const std::string& name) {
    for (const auto& s : sweeps())
        if (s.name == name) return &s;
    return nullptr;
}

void write_csv(const std::vector<Row>& rows, const std::string& path) {
    if (rows.empty()) return;
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

    // Union of keys in first-seen order, so a sweep with extra columns still
    // produces a rectangular file.
    std::vector<std::string> fields;
    for (const auto& row : rows)
        for (const auto& [k, v] : row) {
            (void)v;
            if (std::find(fields.begin(), fields.end(), k) == fields.end()) fields.push_back(k);
        }

    std::ofstream out(path);
    for (std::size_t i = 0; i < fields.size(); ++i) out << (i ? "," : "") << fields[i];
    out << "\n";
    for (const auto& row : rows) {
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i) out << ",";
            for (const auto& [k, v] : row)
                if (k == fields[i]) {
                    out << v;
                    break;
                }
        }
        out << "\n";
    }
    std::cout << "wrote " << rows.size() << " rows -> " << path << std::endl;
}

}  // namespace memora
