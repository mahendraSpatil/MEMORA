#include "memora/tarjan.hpp"

#include <algorithm>

namespace memora {

StructuralIndex::StructuralIndex(const MemoryGraph& g)
    : g_(&g),
      n_(g.n()),
      damage_(static_cast<std::size_t>(g.n()), 0.0),
      comp_of_(static_cast<std::size_t>(g.n()), -1),
      disc_(static_cast<std::size_t>(g.n()), 0),
      low_(static_cast<std::size_t>(g.n()), 0),
      sub_(static_cast<std::size_t>(g.n()), 0),
      seen_(static_cast<std::size_t>(g.n()), 0),
      parent_(static_cast<std::size_t>(g.n()), -1),
      pieces_(static_cast<std::size_t>(g.n())) {}

const std::vector<int>& StructuralIndex::rebuild() {
    std::fill(damage_.begin(), damage_.end(), 0.0);
    std::fill(comp_of_.begin(), comp_of_.end(), -1);
    next_cid_ = 0;
    std::vector<int> roots;
    roots.reserve(static_cast<std::size_t>(n_));
    for (int v = 0; v < n_; ++v)
        if (g_->alive(v)) roots.push_back(v);
    analyse(roots);
    return visited_;
}

const std::vector<int>& StructuralIndex::update_after_delete(int v) {
    damage_[static_cast<std::size_t>(v)] = 0.0;
    comp_of_[static_cast<std::size_t>(v)] = -1;
    std::vector<int> seeds = g_->neighbours(v);
    if (seeds.empty()) {
        visited_.clear();
        return visited_;
    }
    analyse(seeds);
    return visited_;
}

std::vector<int> StructuralIndex::articulation_points() const {
    std::vector<int> out;
    for (int v = 0; v < n_; ++v)
        if (g_->alive(v) && damage_[static_cast<std::size_t>(v)] > 0.0) out.push_back(v);
    return out;
}

void StructuralIndex::analyse(const std::vector<int>& roots) {
    ++analysis_calls;
    ++gen_;
    visited_.clear();

    for (int root : roots) {
        if (!g_->alive(root) || seen_[static_cast<std::size_t>(root)] == gen_) continue;
        dfs(root);

        const int cid = next_cid_++;
        const long long nc = static_cast<long long>(order_.size());
        // C(n-1, 2): the pairs that survive v's removal if nothing splits.
        const long long denom = (nc - 1) * (nc - 2) / 2;

        for (int u : order_) {
            comp_of_[static_cast<std::size_t>(u)] = cid;
            auto& ps = pieces_[static_cast<std::size_t>(u)];
            if (ps.empty()) {
                damage_[static_cast<std::size_t>(u)] = 0.0;
                continue;
            }
            long long kept = 0, total = 0;
            for (int c : ps) {
                kept += static_cast<long long>(c) * (c - 1) / 2;
                total += c;
            }
            const long long rest = nc - 1 - total;
            if (rest > 0) kept += rest * (rest - 1) / 2;
            damage_[static_cast<std::size_t>(u)] = static_cast<double>(denom - kept);
        }
        visited_.insert(visited_.end(), order_.begin(), order_.end());
    }
    nodes_visited += static_cast<long long>(visited_.size());
}

void StructuralIndex::dfs(int root) {
    order_.clear();
    stack_v_.clear();
    stack_i_.clear();

    std::int64_t timer = 0;
    seen_[static_cast<std::size_t>(root)] = gen_;
    disc_[static_cast<std::size_t>(root)] = timer;
    low_[static_cast<std::size_t>(root)] = timer;
    ++timer;
    sub_[static_cast<std::size_t>(root)] = 1;
    parent_[static_cast<std::size_t>(root)] = -1;
    pieces_[static_cast<std::size_t>(root)].clear();
    order_.push_back(root);
    stack_v_.push_back(root);
    stack_i_.push_back(0);

    while (!stack_v_.empty()) {
        const int v = stack_v_.back();
        const auto& edges = g_->adj(v);
        bool descended = false;

        while (stack_i_.back() < static_cast<int>(edges.size())) {
            const int w = edges[static_cast<std::size_t>(stack_i_.back())].to;
            ++stack_i_.back();
            if (!g_->alive(w)) continue;

            if (seen_[static_cast<std::size_t>(w)] != gen_) {
                seen_[static_cast<std::size_t>(w)] = gen_;
                disc_[static_cast<std::size_t>(w)] = timer;
                low_[static_cast<std::size_t>(w)] = timer;
                ++timer;
                sub_[static_cast<std::size_t>(w)] = 1;
                parent_[static_cast<std::size_t>(w)] = v;
                pieces_[static_cast<std::size_t>(w)].clear();
                order_.push_back(w);
                stack_v_.push_back(w);
                stack_i_.push_back(0);
                descended = true;
                break;
            }
            if (w != parent_[static_cast<std::size_t>(v)] &&
                disc_[static_cast<std::size_t>(w)] < low_[static_cast<std::size_t>(v)]) {
                low_[static_cast<std::size_t>(v)] = disc_[static_cast<std::size_t>(w)];
            }
        }
        if (descended) continue;

        stack_v_.pop_back();
        stack_i_.pop_back();
        const int p = parent_[static_cast<std::size_t>(v)];
        if (p != -1) {
            sub_[static_cast<std::size_t>(p)] += sub_[static_cast<std::size_t>(v)];
            if (low_[static_cast<std::size_t>(v)] < low_[static_cast<std::size_t>(p)])
                low_[static_cast<std::size_t>(p)] = low_[static_cast<std::size_t>(v)];
            // The subtree under v cannot reach above p, so removing p cuts it
            // loose: record its size as one of p's separated pieces.
            if (low_[static_cast<std::size_t>(v)] >= disc_[static_cast<std::size_t>(p)])
                pieces_[static_cast<std::size_t>(p)].push_back(
                    static_cast<int>(sub_[static_cast<std::size_t>(v)]));
        }
    }
}

}  // namespace memora
