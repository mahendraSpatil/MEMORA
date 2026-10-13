# Memora — Context-Aware Memory Eviction

A C++17 implementation of CAFA, a memory eviction policy that decides what to
forget by estimating how much **contextual connectivity** each deletion would
destroy — not just how old or how rarely used a memory is.

No external dependencies. `g++` and `make` are enough.

```bash
make            # builds bin/memora and bin/memora_tests
make test       # validation suite (Tarjan vs brute force, DP vs exhaustive, ...)
make demo       # one configuration, table on stdout
make results    # every sweep -> results/*.csv   (~30 s, includes V = 10,000)
make figures    # charts from the CSVs           (needs python3 + matplotlib)
```

---

## What this does differently from the original proposal

Three changes to the design, each of which affects the results.

### 1. Eviction is a ratio, not a weighted sum

The proposal scored `F(m) = w1·A + w2·S + w3(1−I) + w4(1−R) + w5(1−C)`, which
adds size to the value terms — it treats "this file is big" as evidence the
memory is worthless. It is not. Size is the *benefit* of deleting; the other
terms are the *cost*. Mixed into one sum, the weights mean nothing.

Stated properly the task is: free at least B bytes while losing as little
value as possible. That is a **minimum-cost covering knapsack**, which is
NP-hard — which is what justifies a greedy in the first place. The greedy key
is value per byte:

```
V(m)   = (1 − w_C) · [importance, usage, freshness]  +  w_C · C(m)
rho(m) = V(m) / size(m)               evict in increasing rho
```

Size needs no weight of its own, dropping five tunable numbers to four, and
`w_C = 0` gives the ablation for free. A four-line **repair pass** then walks
the deleted list in decreasing value and restores anything the budget did not
need; it can never make the result worse.

Everything is rank-normalised, not min–max normalised: min–max lets one 2 GB
outlier crush every other size to ~0.

### 2. Structural importance is graded, not a yes/no flag

`is_articulation_point` cannot rank. Severing a single leaf and halving a
1,000-node cluster are both "true". So the same Tarjan DFS also computes how
many connected pairs each removal destroys:

```
D(v) = C(n−1, 2) − Σ C(c_i, 2)
```

Subtree sizes fall out of the DFS for free, so this is still **O(V+E) in one
pass**. Useful invariant, asserted in the test suite: `D(v) > 0` exactly when
`v` is an articulation point.

The DFS is iterative. A recursive one blows the stack on a long chain at
V = 10,000.

### 3. The baseline that actually tests the claim

Comparing CAFA against FIFO/LRU/LFU only proves that reading importance beats
ignoring it, which nobody doubts and which has nothing to do with the graph.
So the registry includes **`value_greedy`** — CAFA with `w_C = 0`, the
identical policy with the graph switched off — plus `degree_cafa`, which uses
weighted degree instead of Tarjan. The headline claim of this project is the
gap between two adjacent rows, and the boundary at which it disappears.

---

## Results

Pairwise connectivity retention among survivors, 1,500 memories, 25% of bytes
freed, mean of 10 seeds. Every policy frees the same bytes from the same input.

| avg degree | LRU | value_greedy | cafa_local | cafa_repair | **ablation gap** |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.095 | 0.538 | 0.763 | 0.997 | **+22.5 pts** |
| 3 | 0.649 | 0.898 | 0.949 | 0.998 | **+5.1 pts** |
| 4 | 0.796 | 0.945 | 0.975 | 0.999 | **+3.0 pts** |
| 6 | 0.911 | 0.978 | 0.991 | 0.999 | **+1.3 pts** |
| 8 | 0.954 | 0.989 | 0.996 | 1.000 | **+0.7 pts** |
| 12 | 0.989 | 0.997 | 0.999 | 1.000 | **+0.2 pts** |

**The three findings.**

1. **Structural awareness pays, and the payment is measurable.** On sparse
   graphs CAFA keeps 22 points more connectivity than an otherwise identical
   policy with the graph term removed.
2. **The advantage has a boundary, and it is density.** By average degree 12
   the gap is 0.2 points. Dense graphs survive almost any deletion, so there
   is nothing for a structural term to protect. Reporting that is what makes
   the rest of the numbers credible.
3. **Repair beats scoring.** Reconnecting a deleted memory's surviving
   neighbours (CAFA-R, ~20 lines, O(deg²) per deletion) recovers more context
   than any amount of weight tuning — 0.997 even at degree 2.

Two further results worth a slide:

- **Greedy is near-optimal.** Against an exact O(V·B) knapsack DP at V = 120,
  the greedy lands **2.7%** above optimal for `value_greedy` and **3.8%** for
  `cafa_local` (worst case 8.1%).
- **The incremental update is mostly not worth its cost.** `cafa_local`
  re-runs Tarjan on the affected component after every deletion and beats the
  score-once `cafa_static` by roughly 1.5 points — for **750× the runtime** at
  V = 10,000. That is a real engineering trade-off with a real answer, and the
  honest recommendation is `cafa_static` unless the graph is very sparse.

### Runtime (milliseconds, mean of 3 seeds, 25% budget)

| V | LRU | value_greedy | cafa_static | cafa_local | graph build |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.08 | 0.04 | 0.13 | 9.1 | 5.3 |
| 5,000 | 0.48 | 0.22 | 0.76 | 309 | 69 |
| 10,000 | 1.01 | 0.39 | 1.80 | 1,356 | 230 |

A naive similarity build is O(V²) — 5×10⁷ comparisons at V = 10,000, which
would cost more than every eviction run put together. Blocking by category and
sampling a fixed number of cross-category candidates brings it to 230 ms, at
which point the incremental Tarjan, not construction, is the bottleneck for
`cafa_local`. For every other policy construction still dominates, which is
why it is timed and reported separately.

---

## Layout

```
include/memora/     headers, each one documenting why it is shaped that way
src/
  memory.cpp        memory records and column views
  generator.cpp     planted-cluster synthetic data, with bridge memories
  graph.cpp         adjacency, deletion by flag, bucketed similarity build
  tarjan.cpp        iterative Tarjan + pair damage + incremental update
  scoring.cpp       rank normalisation, retention value, value density
  policies.cpp      random/FIFO/LRU/LFU/size, the ablations, CAFA
  metrics.cpp       survivor-relative retention, held-out context queries
  knapsack.cpp      exact DP reference and brute force
  benchmark.cpp     sweeps -> CSV
  main.cpp          CLI
tests/test_main.cpp validation suite (no framework, plain assertions)
tools/plot.py       figures from results/*.csv (matplotlib only)
reference/python/   the Python prototype, kept as a second implementation
```

### Data structures, and where they are

| Structure | Where | Why |
|---|---|---|
| Adjacency list | `graph.hpp` | O(V+E) space, cache-friendly traversal |
| Doubly linked list + index | `LRUList`, `policies.cpp` | O(1) touch and O(1) tail eviction |
| Binary min-heap, lazy deletion | `cafa()`, `policies.cpp` | `std::priority_queue` has no decrease-key |
| DFS stack (explicit) | `tarjan.cpp` | iterative Tarjan, O(V+E) |
| Union of hash sets | `build_graph`, `graph.cpp` | candidate edge dedup |
| 1-D DP table | `knapsack.cpp` | O(V·B) exact reference |

---

## Honest caveats

Read these before defending the project; someone will ask.

- **CAFA-R's connectivity numbers include repair edges.** The bridges it adds
  are real links in the running system, but they are synthetic relationships
  that did not exist before. `neighbourhood_retention` deliberately uses the
  original adjacency on both sides so repair cannot inflate it.
- **`size_greedy` is a surprisingly strong structural baseline** because it
  meets the budget in the fewest deletions and therefore touches the fewest
  nodes. It pays for that in importance retained. Look at `fig3_pareto.png`,
  not at any single column.
- **Importance is generated, then read by the policies.** That is why
  `true_importance` is held out from them and why the `noise` sweep exists.
  With `importance_noise = 0` the policies see ground truth, which is the
  optimistic case.
- **CAFA optimises pair damage and `pairwise_retention` measures pair
  damage.** `context_query_success` exists to break that circularity: pairs
  are sampled from the intact graph before any deletion and CAFA never
  optimises them.
- **Everything here is synthetic.** The generator plants the cluster structure
  the method is designed to exploit. A real graph (a citation network, a
  message thread graph) would make the claim considerably stronger.

## Reproducing

Every run is seeded and every sweep writes its full configuration into the
CSV, so `results/*.csv` plus this repository regenerate every number and every
figure:

```bash
make clean && make && make test && make results && make figures
```

The Python package under `reference/python/` is a second, independent
implementation of the same algorithms; its Tarjan was validated against
NetworkX's `articulation_points` on 200 random graphs, and the C++ version is
validated against a brute-force recomputation of connectivity on 300 more.
Keeping both is how the structural core was checked without trusting either
one on its own.

## Deferred to future work

Named deliberately — knowing where the boundary of the contribution is, is
part of the contribution.

- Fully dynamic biconnectivity (Holm–de Lichtenberg–Thorup) for sub-linear
  incremental updates, instead of re-running Tarjan on a component
- Learned or adaptive weights rather than a fixed, hand-set `w_C`
- Semantic edges from sentence embeddings instead of category similarity
- Redundancy detection and deduplication before eviction
- Modelling the cost of *regenerating* a deleted memory, not only losing it
- Multi-objective Pareto optimisation instead of a scalarised value function
- Edge-weighted minimum cuts rather than unweighted articulation points
