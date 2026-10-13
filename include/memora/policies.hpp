// Eviction policies: classical baselines, the ablations, and CAFA.
//
// The one rule that makes the comparison meaningful: every policy must free
// at least the same number of bytes from the same input. Otherwise CAFA can
// look good purely by deleting five large items where LRU deleted forty small
// ones, and the results table measures nothing. Every result therefore
// reports bytes_freed and overshoot so the fairness of a run is visible.
//
// Two of the baselines are not classical, and they are the important ones:
//
//   size_greedy   shows how much of CAFA's advantage is just "delete big
//                 things".
//   value_greedy  is CAFA with w_C = 0 -- the identical policy with the graph
//                 switched off. The headline claim of this project is the gap
//                 between that row and the CAFA row. Without it, comparing
//                 against FIFO/LRU/LFU only shows that reading importance
//                 beats ignoring it, which nobody doubts and which has
//                 nothing to do with the graph.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "memora/graph.hpp"
#include "memora/memory.hpp"
#include "memora/scoring.hpp"

namespace memora {

struct EvictionResult {
    std::string policy;
    std::vector<int> deleted;
    double bytes_freed = 0.0;
    double required_mb = 0.0;
    double runtime_s = 0.0;
    int restored = 0;
    long long bridges_added = 0;
    long long analysis_calls = 0;
    long long nodes_visited = 0;

    double overshoot() const {
        return required_mb > 0.0 ? bytes_freed / required_mb : 1.0;
    }
    int n_deleted() const { return static_cast<int>(deleted.size()); }
};

enum class Mode { Static, Local };

struct PolicyParams {
    double w_context = 0.30;
    double cut_weight = 0.70;
    Transform transform = Transform::Sqrt;
    Mode mode = Mode::Local;
    bool bridge_repair = false;
    double bridge_threshold = 0.25;
    bool restore_overshoot = true;
    std::uint64_t seed = 0;
    const std::vector<double>* static_part = nullptr;  // optional, cached
};

// -- classical baselines --------------------------------------------------
EvictionResult policy_random(const Store&, MemoryGraph&, double required_mb,
                             const PolicyParams&);
EvictionResult policy_fifo(const Store&, MemoryGraph&, double required_mb,
                           const PolicyParams&);
EvictionResult policy_lru(const Store&, MemoryGraph&, double required_mb,
                          const PolicyParams&);
EvictionResult policy_lfu(const Store&, MemoryGraph&, double required_mb,
                          const PolicyParams&);
EvictionResult policy_size_greedy(const Store&, MemoryGraph&, double required_mb,
                                  const PolicyParams&);

// -- CAFA and its ablations ----------------------------------------------
EvictionResult cafa(const Store&, MemoryGraph&, double required_mb,
                    const PolicyParams&, const std::string& name);

EvictionResult policy_value_greedy(const Store&, MemoryGraph&, double required_mb,
                                   const PolicyParams&);
EvictionResult policy_degree_cafa(const Store&, MemoryGraph&, double required_mb,
                                  const PolicyParams&);
EvictionResult policy_cafa_static(const Store&, MemoryGraph&, double required_mb,
                                  const PolicyParams&);
EvictionResult policy_cafa_local(const Store&, MemoryGraph&, double required_mb,
                                 const PolicyParams&);
EvictionResult policy_cafa_repair(const Store&, MemoryGraph&, double required_mb,
                                  const PolicyParams&);

using PolicyFn = EvictionResult (*)(const Store&, MemoryGraph&, double,
                                    const PolicyParams&);

// Registry, weakest to strongest. Adding a policy here is all it takes for it
// to appear in every sweep and every figure.
const std::vector<std::pair<std::string, PolicyFn>>& policy_registry();
PolicyFn find_policy(const std::string& name);

// -- LRU's data structure, exposed because the report needs to show it -----
//
// Doubly linked list + hashmap: O(1) touch, O(1) evict from the tail. The
// benchmark evicts from a static snapshot so nothing is touched during a run,
// but touch() is implemented because the O(1) move-to-front is the whole
// reason LRU uses this structure.
class LRUList {
public:
    void push_front(int key);
    void touch(int key);
    int pop_lru();  // -1 when empty
    bool empty() const { return head_ == -1; }

private:
    struct Node {
        int key = -1;
        int prev = -1;
        int next = -1;
    };
    std::vector<Node> nodes_;
    std::vector<int> index_;  // key -> node slot, -1 if absent
    int head_ = -1;
    int tail_ = -1;
    void unlink(int slot);
};

}  // namespace memora
