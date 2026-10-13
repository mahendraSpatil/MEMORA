// Iterative Tarjan with graded pair-connectivity damage.
//
// An articulation point flag is a yes/no answer, and a yes/no answer cannot
// rank. Removing a cut vertex that splits off one leaf costs one connected
// pair; removing one that halves a 1,000-node cluster costs about 250,000. A
// fixed "articulation bonus" treats those as the same event, which is how a
// context-aware policy ends up making visibly silly choices.
//
// So this computes, in the *same* DFS that finds the cut vertices, how many
// connected pairs each removal would destroy:
//
//     D(v) = C(n-1, 2) - sum_i C(c_i, 2)
//
// where c_i are the sizes of the components left behind when v is removed
// from its component of size n. Subtree sizes fall out of the DFS for free,
// so this is still O(V + E) and still one pass.
//
// Useful invariant, checked in tests: D(v) > 0 exactly when v is an
// articulation point.
//
// The DFS is iterative. A recursive one blows the stack on a long chain at
// V = 10,000 in a default 8 MB thread stack, and the crash looks like
// corruption rather than an error.
#pragma once

#include <cstdint>
#include <vector>

#include "memora/graph.hpp"

namespace memora {

// Maintains per-node structural damage and component membership.
//
// rebuild()             analyses the whole live graph in O(V + E).
// update_after_delete() re-analyses only the component v was in, which is
//                       what makes CAFA-Local affordable: components fragment
//                       as the run proceeds, so the cost per update falls
//                       away quickly.
class StructuralIndex {
public:
    explicit StructuralIndex(const MemoryGraph& g);

    // Analyse every live node. Returns the nodes visited.
    const std::vector<int>& rebuild();

    // Call *after* graph.kill(v). Returns the nodes whose damage changed, so
    // the caller can re-push exactly those into its heap.
    const std::vector<int>& update_after_delete(int v);

    const std::vector<double>& damage() const { return damage_; }
    const std::vector<int>& component_of() const { return comp_of_; }

    std::vector<int> articulation_points() const;

    long long analysis_calls = 0;
    long long nodes_visited = 0;

private:
    const MemoryGraph* g_;
    int n_;
    std::vector<double> damage_;
    std::vector<int> comp_of_;
    std::vector<std::int64_t> disc_, low_, sub_, seen_;
    std::vector<int> parent_;
    std::int64_t gen_ = 0;
    int next_cid_ = 0;

    std::vector<int> visited_;   // scratch, returned by reference
    std::vector<int> order_;     // scratch, one component
    std::vector<int> stack_v_;   // explicit DFS stack: node
    std::vector<int> stack_i_;   // explicit DFS stack: next edge index
    std::vector<std::vector<int>> pieces_;  // node -> separated subtree sizes

    void analyse(const std::vector<int>& roots);
    void dfs(int root);
};

}  // namespace memora
