/* ═══════════════════════════════════════════════════════════════
   MEMORA UI — Next-Gen Interactive Graph Engine & Controls
   Client-Side Engine for GitHub Pages & Local Deployment
   ═══════════════════════════════════════════════════════════════ */

// ──────────────── Palette & Configurations ────────────────
const CAT_COLORS = [
  "#6366f1", "#06b6d4", "#f59e0b", "#10b981", "#ec4899",
  "#8b5cf6", "#14b8a6", "#f97316", "#84cc16", "#e11d48",
  "#3b82f6", "#a855f7", "#22d3ee", "#eab308", "#d946ef",
];

const POLICY_LABELS = {
  random: "Random",
  fifo: "FIFO",
  lru: "LRU",
  lfu: "LFU",
  size_greedy: "Size Greedy",
  value_greedy: "Value Greedy",
  degree_cafa: "Degree CAFA",
  cafa_static: "CAFA Static",
  cafa_local: "CAFA Local",
  cafa_repair: "CAFA Repair",
};

const POLICY_COLORS = {
  random: "#64748b",
  fifo: "#94a3b8",
  lru: "#f59e0b",
  lfu: "#f97316",
  size_greedy: "#ef4444",
  value_greedy: "#10b981",
  degree_cafa: "#06b6d4",
  cafa_static: "#3b82f6",
  cafa_local: "#6366f1",
  cafa_repair: "#8b5cf6",
};

const ALL_POLICIES = [
  "random", "fifo", "lru", "lfu", "size_greedy",
  "value_greedy", "degree_cafa", "cafa_static", "cafa_local", "cafa_repair",
];

const TOPICS = [
  "lecture notes", "assignment", "exam paper", "lab report", "meeting notes",
  "screenshot", "invoice", "photo", "chat log", "bookmark", "email thread",
  "code snippet", "reference paper", "todo list", "contact card",
];

// ──────────────── Pure JS Pseudo-Random Number Generator ────────────────
class PRNG {
  constructor(seed = 42) {
    this.s = seed | 0;
  }
  next() {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(min = 0, max = 1) {
    return min + (max - min) * this.next();
  }
  normal(mean = 0, std = 1) {
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    return mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  lognormal(mu = 0.9, sigma = 0.9) {
    return Math.exp(this.normal(mu, sigma));
  }
  beta(alpha, beta) {
    const gamma = (a) => {
      if (a < 1) return gamma(a + 1) * Math.pow(this.next(), 1 / a);
      const d = a - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      while (true) {
        let z = this.normal();
        let v = 1 + c * z;
        if (v <= 0) continue;
        v = v * v * v;
        let u = this.next();
        if (u < 1 - 0.0331 * z * z * z * z) return d * v;
        if (Math.log(u) < 0.5 * z * z + d * (1 - v + Math.log(v))) return d * v;
      }
    };
    const x = gamma(alpha);
    const y = gamma(beta);
    return x / (x + y);
  }
  poisson(lambda = 1) {
    const L = Math.exp(-lambda);
    let k = 0, p = 1;
    do {
      k++;
      p *= this.next();
    } while (p > L);
    return k - 1;
  }
}

// ──────────────── Math & Stats Utilities ────────────────
function rankNormalize(arr) {
  const n = arr.length;
  if (n < 2) return new Float64Array(n);
  const indexed = [];
  for (let i = 0; i < n; i++) indexed.push({ val: arr[i], idx: i });
  indexed.sort((a, b) => a.val - b.val);

  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && indexed[j + 1].val === indexed[i].val) j++;
    const avg = 0.5 * (i + j);
    for (let k = i; k <= j; k++) ranks[indexed[k].idx] = avg;
    i = j + 1;
  }
  for (let k = 0; k < n; k++) ranks[k] /= (n - 1);
  return ranks;
}

// ──────────────── Client-Side Graph Models ────────────────
class MemoryStore {
  constructor(memories) {
    this.memories = memories;
    this.n = memories.length;
    this.total_size_mb = memories.reduce((acc, m) => acc + m.size_mb, 0);
  }
  similarity(i, j) {
    const vi = this.memories[i].vector;
    const vj = this.memories[j].vector;
    let dot = 0;
    for (let k = 0; k < vi.length; k++) dot += vi[k] * vj[k];
    return dot;
  }
}

class MemoryGraph {
  constructor(n) {
    this.n = n;
    this.adj = Array.from({ length: n }, () => new Map());
    this.alive = new Uint8Array(n).fill(1);
    this._orig_comp = null;
  }
  addEdge(u, v, w = 1.0) {
    if (u === v) return;
    this.adj[u].set(v, w);
    this.adj[v].set(u, w);
  }
  neighbours(v) {
    const res = [];
    for (const [u] of this.adj[v]) if (this.alive[u]) res.push(u);
    return res;
  }
  degree(v) {
    let d = 0;
    for (const [u] of this.adj[v]) if (this.alive[u]) d++;
    return d;
  }
  weightedDegree(v) {
    let wd = 0;
    for (const [u, w] of this.adj[v]) if (this.alive[u]) wd += w;
    return wd;
  }
  edgeCount() {
    let count = 0;
    for (let i = 0; i < this.n; i++) {
      if (!this.alive[i]) continue;
      for (const [j] of this.adj[i]) {
        if (this.alive[j] && j > i) count++;
      }
    }
    return count;
  }
  delete(v) { this.alive[v] = 0; }
  restore(v) { this.alive[v] = 1; }
  clone() {
    const g = new MemoryGraph(this.n);
    for (let i = 0; i < this.n; i++) {
      for (const [u, w] of this.adj[i]) g.adj[i].set(u, w);
    }
    g.alive.set(this.alive);
    g._orig_comp = this._orig_comp;
    return g;
  }
  originalComponents() {
    if (this._orig_comp) return this._orig_comp;
    const comp = new Int32Array(this.n).fill(-1);
    let cid = 0;
    for (let start = 0; start < this.n; start++) {
      if (comp[start] !== -1) continue;
      const stack = [start];
      comp[start] = cid;
      while (stack.length) {
        const v = stack.pop();
        for (const [w] of this.adj[v]) {
          if (comp[w] === -1) {
            comp[w] = cid;
            stack.push(w);
          }
        }
      }
      cid++;
    }
    this._orig_comp = comp;
    return comp;
  }
  connectedComponents() {
    const comp = new Int32Array(this.n).fill(-1);
    const sizes = {};
    let cid = 0;
    for (let start = 0; start < this.n; start++) {
      if (!this.alive[start] || comp[start] !== -1) continue;
      const stack = [start];
      comp[start] = cid;
      let count = 0;
      while (stack.length) {
        const v = stack.pop();
        count++;
        for (const [w] of this.adj[v]) {
          if (this.alive[w] && comp[w] === -1) {
            comp[w] = cid;
            stack.push(w);
          }
        }
      }
      sizes[cid] = count;
      cid++;
    }
    return { comp, sizes };
  }
}

// ──────────────── Iterative Tarjan Structural Index ────────────────
class StructuralIndex {
  constructor(graph) {
    this.g = graph;
    this.n = graph.n;
    this.damage = new Float64Array(this.n);
    this.comp_of = new Int32Array(this.n).fill(-1);
    this.comp_size = {};
    this._disc = new Int32Array(this.n);
    this._low = new Int32Array(this.n);
    this._sub = new Int32Array(this.n);
    this._seen = new Int32Array(this.n);
    this._parent = new Int32Array(this.n).fill(-1);
    this._gen = 0;
    this._next_cid = 0;
  }
  rebuild() {
    this.comp_size = {};
    this._next_cid = 0;
    this.comp_of.fill(-1);
    this.damage.fill(0);
    const roots = [];
    for (let i = 0; i < this.n; i++) if (this.g.alive[i]) roots.push(i);
    return this._analyse(roots);
  }
  updateAfterDelete(v) {
    this.damage[v] = 0;
    this.comp_of[v] = -1;
    const seeds = this.g.neighbours(v);
    if (!seeds.length) return [];
    return this._analyse(seeds);
  }
  _analyse(roots) {
    this._gen++;
    const gen = this._gen;
    const visited = [];
    for (const root of roots) {
      if (!this.g.alive[root] || this._seen[root] === gen) continue;
      const { order, pieces } = this._dfs(root, gen);
      const cid = this._next_cid++;
      const n_comp = order.length;
      this.comp_size[cid] = n_comp;
      const denom = ((n_comp - 1) * (n_comp - 2)) / 2;
      for (const v of order) {
        this.comp_of[v] = cid;
        const ps = pieces[v];
        if (ps && ps.length) {
          let kept = 0;
          let total = 0;
          for (const c of ps) {
            kept += (c * (c - 1)) / 2;
            total += c;
          }
          const rest = n_comp - 1 - total;
          if (rest > 0) kept += (rest * (rest - 1)) / 2;
          this.damage[v] = Math.max(0, denom - kept);
        } else {
          this.damage[v] = 0;
        }
      }
      for (const o of order) visited.push(o);
    }
    return visited;
  }
  _dfs(root, gen) {
    const pieces = {};
    const order = [root];
    let timer = 0;
    this._seen[root] = gen;
    this._disc[root] = this._low[root] = timer++;
    this._sub[root] = 1;
    this._parent[root] = -1;

    const rootNeighbours = [];
    for (const [w] of this.g.adj[root]) if (this.g.alive[w]) rootNeighbours.push(w);
    const stack = [[root, rootNeighbours, 0]];

    while (stack.length) {
      const top = stack[stack.length - 1];
      const v = top[0];
      const neigh = top[1];
      let idx = top[2];

      let descended = false;
      while (idx < neigh.length) {
        const w = neigh[idx++];
        top[2] = idx;
        if (!this.g.alive[w]) continue;
        if (this._seen[w] !== gen) {
          this._seen[w] = gen;
          this._disc[w] = this._low[w] = timer++;
          this._sub[w] = 1;
          this._parent[w] = v;
          order.push(w);
          const wNeigh = [];
          for (const [n] of this.g.adj[w]) if (this.g.alive[n]) wNeigh.push(n);
          stack.push([w, wNeigh, 0]);
          descended = true;
          break;
        }
        if (w !== this._parent[v] && this._disc[w] < this._low[v]) {
          this._low[v] = this._disc[w];
        }
      }
      if (descended) continue;

      stack.pop();
      const p = this._parent[v];
      if (p !== -1) {
        this._sub[p] += this._sub[v];
        if (this._low[v] < this._low[p]) this._low[p] = this._low[v];
        if (this._low[v] >= this._disc[p]) {
          if (!pieces[p]) pieces[p] = [];
          pieces[p].push(this._sub[v]);
        }
      }
    }
    return { order, pieces };
  }
}

// ──────────────── Generator & Builder ────────────────
function generateStore(n = 150, nCategories = 8, bridgeFraction = 0.05, seed = 42) {
  const rng = new PRNG(seed);
  const dim = 16;
  const centroids = Array.from({ length: nCategories }, () => {
    const v = Array.from({ length: dim }, () => rng.normal());
    const mag = Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
    return v.map((x) => x / mag);
  });

  const nBridges = Math.round(n * bridgeFraction);
  const bridgeIds = new Set();
  while (bridgeIds.size < nBridges) bridgeIds.add(Math.floor(rng.uniform(0, n)));

  const memories = [];
  for (let i = 0; i < n; i++) {
    const cat = Math.floor(rng.uniform(0, nCategories));
    let base;
    if (bridgeIds.has(i) && nCategories > 1) {
      let other = Math.floor(rng.uniform(0, nCategories - 1));
      if (other >= cat) other++;
      const mix = rng.uniform(0.35, 0.65);
      base = centroids[cat].map((val, d) => mix * val + (1 - mix) * centroids[other][d]);
    } else {
      base = centroids[cat];
    }
    const vec = base.map((val) => val + 0.55 * rng.normal());
    const mag = Math.sqrt(vec.reduce((a, b) => a + b * b, 0)) || 1;
    const normVec = vec.map((x) => x / mag);

    const size_mb = Math.max(0.1, Math.min(400.0, rng.lognormal(0.9, 0.9)));
    const true_imp = rng.beta(2.0, 3.0);
    const obs_imp = true_imp;
    const age = rng.uniform(0, 365);
    const recency = age * rng.beta(1.0, 1.0 + 3.0 * true_imp);
    const lam = 1.0 + 25.0 * true_imp * Math.exp(-recency / 60.0);
    const access = rng.poisson(lam);

    memories.push({
      id: i,
      name: `${TOPICS[cat % TOPICS.length]} #${i}`,
      category: cat,
      size_mb: Math.round(size_mb * 1000) / 1000,
      true_importance: Math.round(true_imp * 1000) / 1000,
      observed_importance: Math.round(obs_imp * 1000) / 1000,
      age_days: Math.round(age * 10) / 10,
      recency_days: Math.round(recency * 10) / 10,
      access_count: access,
      vector: normVec,
      is_bridge: bridgeIds.has(i),
    });
  }
  return new MemoryStore(memories);
}

function buildGraph(store, targetDegree = 4.0, seed = 42) {
  const n = store.n;
  const g = new MemoryGraph(n);
  const rng = new PRNG(seed);
  const k = Math.max(2, Math.round(targetDegree) + 2);

  const buckets = {};
  for (let i = 0; i < n; i++) {
    const c = store.memories[i].category;
    if (!buckets[c]) buckets[c] = [];
    buckets[c].push(i);
  }

  const cand = new Map();
  const bestOf = new Map();

  function offer(i, j, s) {
    if (i === j || s < 0.15) return;
    const u = Math.min(i, j);
    const v = Math.max(i, j);
    const key = `${u},${v}`;
    if (s > (cand.get(key) || -1)) cand.set(key, s);
    if (!bestOf.has(i) || s > bestOf.get(i).s) bestOf.set(i, { s, j });
    if (!bestOf.has(j) || s > bestOf.get(j).s) bestOf.set(j, { s, i });
  }

  for (const members of Object.values(buckets)) {
    for (let a = 0; a < members.length; a++) {
      const i = members[a];
      const sims = [];
      for (let b = 0; b < members.length; b++) {
        if (a === b) continue;
        const j = members[b];
        sims.push({ j, s: store.similarity(i, j) });
      }
      sims.sort((x, y) => y.s - x.s);
      for (let t = 0; t < Math.min(k, sims.length); t++) {
        offer(i, sims[t].j, sims[t].s);
      }
    }
  }

  const crossSamples = 32;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < crossSamples; c++) {
      const j = Math.floor(rng.uniform(0, n));
      offer(i, j, store.similarity(i, j));
    }
  }

  const chosen = new Set();
  for (const [i, { s, j }] of bestOf.entries()) {
    if (j >= 0) {
      const u = Math.min(i, j);
      const v = Math.max(i, j);
      chosen.add(`${u},${v}`);
    }
  }

  const budget = Math.round((n * targetDegree) / 2);
  if (chosen.size < budget) {
    const rest = [];
    for (const [key, s] of cand.entries()) {
      if (!chosen.has(key)) rest.push({ key, s });
    }
    rest.sort((a, b) => b.s - a.s);
    for (let i = 0; i < Math.min(budget - chosen.size, rest.length); i++) {
      chosen.add(rest[i].key);
    }
  }

  for (const edgeKey of chosen) {
    const [u, v] = edgeKey.split(",").map(Number);
    g.addEdge(u, v, cand.get(edgeKey) || 1.0);
  }
  return g;
}

// ──────────────── Scoring & Features ────────────────
function staticFeatures(store) {
  const n = store.n;
  const obsImp = store.memories.map(m => m.observed_importance);
  const recency = store.memories.map(m => m.recency_days);
  const access = store.memories.map(m => Math.log1p(m.access_count));
  const age = store.memories.map(m => m.age_days);

  const impR = rankNormalize(obsImp);
  const recR = rankNormalize(recency);
  const accR = rankNormalize(access);
  const ageR = rankNormalize(age);

  const combined = new Float64Array(n);
  const b0 = 0.35 / 0.70;
  const b1 = 0.20 / 0.70;
  const b2 = 0.15 / 0.70;

  for (let i = 0; i < n; i++) {
    const usage = 0.5 * (1.0 - recR[i]) + 0.5 * accR[i];
    const fresh = 1.0 - ageR[i];
    combined[i] = b0 * impR[i] + b1 * usage + b2 * fresh;
  }
  return combined;
}

function structuralScore(damage, wdeg, damageScale, maxWdeg, cutWeight = 0.7) {
  const n = damage.length;
  const score = new Float64Array(n);
  const scale = Math.max(1.0, damageScale);
  for (let i = 0; i < n; i++) {
    const d = Math.sqrt(Math.max(0, Math.min(1.0, damage[i] / scale)));
    const loc = maxWdeg > 0 ? Math.min(1.0, wdeg[i] / maxWdeg) : 0;
    score[i] = cutWeight * d + (1.0 - cutWeight) * loc;
  }
  return score;
}

function retentionValue(staticPart, context, wContext) {
  const n = staticPart.length;
  const val = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    val[i] = (1.0 - wContext) * staticPart[i] + wContext * context[i];
  }
  return val;
}

function evaluate(store, original, g, deleted, runtimeMs = 0) {
  const n = store.n;
  const origComp = original.originalComponents();
  const origGroups = {};
  for (let i = 0; i < n; i++) {
    if (g.alive[i]) {
      const cid = origComp[i];
      origGroups[cid] = (origGroups[cid] || 0) + 1;
    }
  }
  const { sizes: newSizes } = g.connectedComponents();

  let pairsBefore = 0;
  for (const c of Object.values(origGroups)) pairsBefore += (c * (c - 1)) / 2;
  let pairsAfter = 0;
  for (const c of Object.values(newSizes)) pairsAfter += (c * (c - 1)) / 2;

  const contextRetention = pairsBefore > 0 ? pairsAfter / pairsBefore : 1.0;

  let trueImpBefore = 0;
  let trueImpAfter = 0;
  for (let i = 0; i < n; i++) {
    trueImpBefore += store.memories[i].true_importance;
    if (g.alive[i]) trueImpAfter += store.memories[i].true_importance;
  }
  const importanceRetention = trueImpBefore > 0 ? trueImpAfter / trueImpBefore : 1.0;

  let freedMb = 0;
  for (const d of deleted) freedMb += store.memories[d].size_mb;

  const bridgesTotal = store.memories.filter(m => m.is_bridge).length;
  const bridgesSurvived = store.memories.filter(m => m.is_bridge && g.alive[m.id]).length;

  return {
    context_retention: contextRetention,
    importance_retention: importanceRetention,
    freed_mb: freedMb,
    bridges_survived: bridgesSurvived,
    bridges_total: bridgesTotal,
    runtime_ms: Math.round(runtimeMs * 10) / 10,
    n_deleted: deleted.length,
  };
}

// ──────────────── Eviction Policies Evaluator ────────────────
function runPoliciesClient(store, original, budgetPct = 25.0, selectedPolicies = ALL_POLICIES) {
  const n = store.n;
  const requiredMb = store.total_size_mb * (budgetPct / 100.0);
  const staticPart = staticFeatures(store);
  const results = [];

  for (const pname of ALL_POLICIES) {
    if (!selectedPolicies.includes(pname)) continue;
    const g = original.clone();
    const t0 = performance.now();
    let deleted = [];
    let freed = 0;

    if (pname === "random") {
      const order = [...Array(n).keys()].sort(() => Math.random() - 0.5);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "fifo") {
      const order = [...Array(n).keys()].sort((a, b) => store.memories[b].age_days - store.memories[a].age_days);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "lru") {
      const order = [...Array(n).keys()].sort((a, b) => store.memories[b].recency_days - store.memories[a].recency_days);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "lfu") {
      const order = [...Array(n).keys()].sort((a, b) => store.memories[a].access_count - store.memories[b].access_count);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "size_greedy") {
      const order = [...Array(n).keys()].sort((a, b) => store.memories[b].size_mb - store.memories[a].size_mb);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "value_greedy") {
      // CAFA with w_C = 0
      const rho = staticPart.map((v, i) => v / store.memories[i].size_mb);
      const order = [...Array(n).keys()].sort((a, b) => rho[a] - rho[b]);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "degree_cafa") {
      // Degree CAFA (w_C = 0.30, cut_weight = 0)
      const wdeg = new Float64Array(n);
      for (let i = 0; i < n; i++) wdeg[i] = g.weightedDegree(i);
      const maxWdeg = Math.max(1.0, ...wdeg);
      const ctx = structuralScore(new Float64Array(n), wdeg, 1.0, maxWdeg, 0.0);
      const val = retentionValue(staticPart, ctx, 0.30);
      const rho = val.map((v, i) => v / store.memories[i].size_mb);
      const order = [...Array(n).keys()].sort((a, b) => rho[a] - rho[b]);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "cafa_static") {
      const si = new StructuralIndex(g);
      si.rebuild();
      const wdeg = new Float64Array(n);
      for (let i = 0; i < n; i++) wdeg[i] = g.weightedDegree(i);
      const maxWdeg = Math.max(1.0, ...wdeg);
      const damageScale = Math.max(1.0, ...si.damage);
      const ctx = structuralScore(si.damage, wdeg, damageScale, maxWdeg, 0.7);
      const val = retentionValue(staticPart, ctx, 0.30);
      const rho = val.map((v, i) => v / store.memories[i].size_mb);
      const order = [...Array(n).keys()].sort((a, b) => rho[a] - rho[b]);
      for (const v of order) {
        if (freed >= requiredMb) break;
        g.delete(v);
        deleted.push(v);
        freed += store.memories[v].size_mb;
      }
    } else if (pname === "cafa_local" || pname === "cafa_repair") {
      const bridgeRepair = pname === "cafa_repair";
      const si = new StructuralIndex(g);
      si.rebuild();
      const wdeg = new Float64Array(n);
      for (let i = 0; i < n; i++) wdeg[i] = g.weightedDegree(i);
      const maxWdeg = Math.max(1.0, ...wdeg);
      const damageScale = Math.max(1.0, ...si.damage);

      const ctx = structuralScore(si.damage, wdeg, damageScale, maxWdeg, 0.7);
      const val = retentionValue(staticPart, ctx, 0.30);
      const rho = val.map((v, i) => v / store.memories[i].size_mb);

      while (freed < requiredMb) {
        let bestV = -1, bestRho = Infinity;
        for (let i = 0; i < n; i++) {
          if (g.alive[i] && rho[i] < bestRho) {
            bestRho = rho[i];
            bestV = i;
          }
        }
        if (bestV < 0) break;
        const neigh = g.neighbours(bestV);
        g.delete(bestV);
        deleted.push(bestV);
        freed += store.memories[bestV].size_mb;

        if (bridgeRepair && neigh.length > 1) {
          for (let a = 0; a < neigh.length; a++) {
            for (let b = a + 1; b < neigh.length; b++) {
              const u = neigh[a], v = neigh[b];
              if (!g.adj[u].has(v)) {
                const s = store.similarity(u, v);
                if (s >= 0.25) g.addEdge(u, v, s);
              }
            }
          }
        }

        const changed = si.updateAfterDelete(bestV);
        if (changed.length) {
          for (const u of changed) wdeg[u] = g.weightedDegree(u);
          const newCtx = structuralScore(si.damage, wdeg, damageScale, maxWdeg, 0.7);
          const newVal = retentionValue(staticPart, newCtx, 0.30);
          for (const u of changed) {
            val[u] = newVal[u];
            rho[u] = newVal[u] / store.memories[u].size_mb;
          }
        }
      }
    }

    const tEnd = performance.now();
    const m = evaluate(store, original, g, deleted, tEnd - t0);
    m.policy = pname;
    results.push(m);
  }

  return { results, required_mb: requiredMb };
}

function graphToJson(graph, store, damage) {
  const nodes = [];
  for (let v = 0; v < graph.n; v++) {
    const node = { id: v, alive: Boolean(graph.alive[v]) };
    if (store) {
      const m = store.memories[v];
      node.name = m.name;
      node.category = m.category;
      node.size_mb = m.size_mb;
      node.true_importance = m.true_importance;
      node.observed_importance = m.observed_importance;
      node.age_days = m.age_days;
      node.recency_days = m.recency_days;
      node.access_count = m.access_count;
      node.is_bridge = m.is_bridge;
    }
    if (damage) node.damage = Math.round(damage[v] * 100) / 100;
    nodes.push(node);
  }

  const edges = [];
  const seen = new Set();
  for (let v = 0; v < graph.n; v++) {
    for (const [w, wt] of graph.adj[v]) {
      const key = v < w ? `${v},${w}` : `${w},${v}`;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push({
          source: Math.min(v, w),
          target: Math.max(v, w),
          weight: Math.round(wt * 1000) / 1000,
        });
      }
    }
  }
  return { nodes, edges };
}

// ──────────────── Global App State ────────────────
let clientStore = null;
let clientGraph = null;
let clientDamage = null;

let graphData = null;
let nodePositions = [];
let hoveredNode = null;
let selectedNode = null;
let showLabels = true;

// Pan & Zoom
let panX = 0, panY = 0, zoom = 1.0;
let isDragging = false, dragStartX = 0, dragStartY = 0;

let stepperPanX = 0, stepperPanY = 0, stepperZoom = 1.0;
let isStepperDragging = false, stepperDragStartX = 0, stepperDragStartY = 0;

// Stepper state
let autoPlayTimer = null;
let autoPlaySpeed = 400;
let stepperGraph = null;
let stepperSI = null;
let stepperStatic = null;
let stepperVal = null;
let stepperRho = null;
let stepperFreed = 0;
let stepperRequired = 0;
let stepperGraphData = null;
let stepperNodePositions = [];
let justEvictedId = null;
let stepEventCount = 0;

// ──────────────── Navigation & Status ────────────────
const statusDot = document.getElementById("system-status-dot");
const statusText = document.getElementById("system-status-text");

function setStatus(text, type = "online") {
  if (statusText) statusText.textContent = text;
  if (statusDot) {
    statusDot.className = "status-pulse-dot";
    if (type === "busy") statusDot.style.background = "#f59e0b";
    else if (type === "error") statusDot.style.background = "#f43f5e";
    else statusDot.style.background = "#10b981";
  }
}

document.querySelectorAll(".nav-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".page-view").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const targetPage = document.getElementById("page-" + btn.dataset.tab);
    if (targetPage) targetPage.classList.add("active");

    if (btn.dataset.tab === "playground") {
      resizeCanvas("graph-canvas");
      renderGraph();
    } else if (btn.dataset.tab === "stepper") {
      resizeCanvas("stepper-canvas");
      renderStepperGraph();
    }
  });
});

// Sliders
function bindSlider(id, valId, fmt) {
  const inp = document.getElementById(id);
  const val = document.getElementById(valId);
  if (!inp || !val) return;
  const update = () => {
    val.textContent = fmt ? fmt(inp.value) : inp.value;
  };
  inp.addEventListener("input", update);
  update();
}
bindSlider("inp-n", "val-n");
bindSlider("inp-categories", "val-categories");
bindSlider("inp-degree", "val-degree", (v) => parseFloat(v).toFixed(1));
bindSlider("inp-bridges", "val-bridges", (v) => Math.round(v * 100) + "%");
bindSlider("inp-budget", "val-budget", (v) => v + "%");
bindSlider("inp-step-budget", "val-step-budget", (v) => v + "%");

document.getElementById("btn-random-seed")?.addEventListener("click", () => {
  document.getElementById("inp-seed").value = Math.floor(Math.random() * 10000);
});

// Policy Checkboxes
const checkboxContainer = document.getElementById("policy-checkboxes");
if (checkboxContainer) {
  ALL_POLICIES.forEach((p) => {
    const lbl = document.createElement("label");
    lbl.className = "policy-chip-label active";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = p;
    cb.checked = true;
    cb.addEventListener("change", () => {
      lbl.classList.toggle("active", cb.checked);
    });
    lbl.appendChild(cb);

    const dot = document.createElement("span");
    dot.className = "policy-color-dot";
    dot.style.background = POLICY_COLORS[p] || "#64748b";
    lbl.appendChild(dot);

    lbl.appendChild(document.createTextNode(" " + (POLICY_LABELS[p] || p)));
    checkboxContainer.appendChild(lbl);
  });
}

function showLoading(text) {
  const overlay = document.getElementById("loading-overlay");
  const loadText = document.getElementById("loading-text");
  if (loadText) loadText.textContent = text || "PROCESSING GRAPH TOPOLOGY…";
  if (overlay) overlay.style.display = "flex";
  setStatus("SIMULATING…", "busy");
}

function hideLoading() {
  const overlay = document.getElementById("loading-overlay");
  if (overlay) overlay.style.display = "none";
  setStatus("SYSTEM ONLINE", "online");
}

// ──────────────── Canvas Scaling & Layout ────────────────
function resizeCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = rect.width || 800;
  const h = rect.height || 600;

  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", () => {
  resizeCanvas("graph-canvas");
  resizeCanvas("stepper-canvas");
  renderGraph();
  renderStepperGraph();
});

function layoutGraph(data, width, height) {
  const nodes = data.nodes;
  const edges = data.edges;
  const n = nodes.length;

  const positions = new Array(n);
  const nCats = Math.max(1, ...nodes.map((nd) => nd.category ?? 0)) + 1;
  const catCenters = [];
  const radius = Math.min(width, height) * 0.32;

  for (let c = 0; c < nCats; c++) {
    const angle = (2 * Math.PI * c) / nCats;
    catCenters.push({
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle),
    });
  }

  for (let i = 0; i < n; i++) {
    const cat = nodes[i].category ?? 0;
    const center = catCenters[cat] || { x: width / 2, y: height / 2 };
    const jitter = 50;
    positions[i] = {
      x: center.x + (Math.random() - 0.5) * jitter,
      y: center.y + (Math.random() - 0.5) * jitter,
      vx: 0,
      vy: 0,
    };
  }

  const iterations = Math.min(120, Math.max(60, Math.floor(10000 / n)));
  const kRepel = 2400;
  const kSpring = 0.04;
  const naturalLength = 45;

  for (let it = 0; it < iterations; it++) {
    const temp = 1.0 - it / iterations;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = positions[i].x - positions[j].x;
        const dy = positions[i].y - positions[j].y;
        const distSq = dx * dx + dy * dy + 1;
        const dist = Math.sqrt(distSq);
        if (dist > 250) continue;
        const force = (kRepel / distSq) * temp;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        positions[i].vx += fx;
        positions[i].vy += fy;
        positions[j].vx -= fx;
        positions[j].vy -= fy;
      }
    }

    for (const e of edges) {
      const u = e.source;
      const v = e.target;
      if (u >= n || v >= n) continue;
      const dx = positions[v].x - positions[u].x;
      const dy = positions[v].y - positions[u].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist - naturalLength) * kSpring * temp;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      positions[u].vx += fx;
      positions[u].vy += fy;
      positions[v].vx -= fx;
      positions[v].vy -= fy;
    }

    for (let i = 0; i < n; i++) {
      const dx = width / 2 - positions[i].x;
      const dy = height / 2 - positions[i].y;
      positions[i].vx += dx * 0.002 * temp;
      positions[i].vy += dy * 0.002 * temp;

      positions[i].x += Math.max(-15, Math.min(15, positions[i].vx));
      positions[i].y += Math.max(-15, Math.min(15, positions[i].vy));
      positions[i].vx *= 0.85;
      positions[i].vy *= 0.85;
    }
  }

  return positions.map((p) => ({ x: p.x, y: p.y }));
}

// ──────────────── Canvas Rendering ────────────────
function renderGraph() {
  const canvas = document.getElementById("graph-canvas");
  if (!canvas || !graphData || !nodePositions.length) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.translate(panX + w / 2, panY + h / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-w / 2, -h / 2);

  const nodes = graphData.nodes;
  const edges = graphData.edges;

  // Draw Edges
  for (const e of edges) {
    const u = e.source;
    const v = e.target;
    if (!nodePositions[u] || !nodePositions[v]) continue;

    const alive = (nodes[u]?.alive ?? true) && (nodes[v]?.alive ?? true);
    const isConnectedToHovered =
      hoveredNode !== null && (hoveredNode.id === u || hoveredNode.id === v);

    ctx.beginPath();
    ctx.moveTo(nodePositions[u].x, nodePositions[u].y);
    ctx.lineTo(nodePositions[v].x, nodePositions[v].y);

    if (isConnectedToHovered) {
      ctx.strokeStyle = "rgba(6, 182, 212, 0.9)";
      ctx.lineWidth = 2.4;
      ctx.shadowColor = "#06b6d4";
      ctx.shadowBlur = 8;
    } else if (alive) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.09)";
      ctx.lineWidth = 1.0;
      ctx.shadowBlur = 0;
    } else {
      ctx.strokeStyle = "rgba(51, 65, 85, 0.25)";
      ctx.lineWidth = 0.8;
      ctx.shadowBlur = 0;
    }
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // Draw Nodes
  for (let i = 0; i < nodes.length; i++) {
    const nd = nodes[i];
    const pos = nodePositions[i];
    if (!pos) continue;

    const isHovered = hoveredNode && hoveredNode.id === nd.id;
    const isSelected = selectedNode && selectedNode.id === nd.id;
    const radius = isHovered ? 8.5 : isSelected ? 8 : nd.is_bridge ? 6.5 : 5;

    if (nd.alive) {
      if (nd.damage && nd.damage > 0) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(244, 63, 94, 0.7)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      } else if (nd.is_bridge) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(139, 92, 246, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(139, 92, 246, 0.8)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

    if (!nd.alive) {
      ctx.fillStyle = "#1e293b";
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1;
    } else {
      const catColor = CAT_COLORS[(nd.category || 0) % CAT_COLORS.length];
      ctx.fillStyle = catColor;
      ctx.strokeStyle = isHovered ? "#fff" : "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = isHovered ? 2.5 : 1.2;
    }
    ctx.fill();
    ctx.stroke();

    if (showLabels && (isHovered || isSelected || nd.is_bridge || (nd.damage && nd.damage > 0))) {
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillStyle = isHovered ? "#38bdf8" : "rgba(241, 245, 249, 0.85)";
      ctx.textAlign = "center";
      ctx.fillText(`#${nd.id}`, pos.x, pos.y - radius - 6);
    }
  }

  ctx.restore();
}

// ──────────────── Node Tooltip & Inspector ────────────────
const tooltip = document.getElementById("graph-tooltip");
const inspector = document.getElementById("node-inspector");

function updateInspector(node) {
  if (!inspector || !node) return;
  selectedNode = node;
  inspector.style.display = "block";

  document.getElementById("insp-name").textContent = node.name || `Memory #${node.id}`;
  document.getElementById("insp-cat-badge").textContent = `CLUSTER ${node.category ?? 0}`;
  document.getElementById("insp-size").textContent = `${node.size_mb ?? 0} MB`;

  const typeEl = document.getElementById("insp-type");
  if (!node.alive) {
    typeEl.textContent = "EVICTED";
    typeEl.style.color = "#94a3b8";
  } else if (node.damage && node.damage > 0) {
    typeEl.textContent = "CUT VERTEX (ARTICULATION)";
    typeEl.style.color = "#f43f5e";
  } else if (node.is_bridge) {
    typeEl.textContent = "BRIDGE NODE";
    typeEl.style.color = "#8b5cf6";
  } else {
    typeEl.textContent = "ACTIVE";
    typeEl.style.color = "#06b6d4";
  }

  const imp = node.observed_importance ?? 0;
  document.getElementById("insp-imp").textContent = imp.toFixed(2);
  document.getElementById("insp-imp-bar").style.width = `${Math.min(100, imp * 100)}%`;

  const dmg = node.damage ?? 0;
  document.getElementById("insp-damage").textContent = `${dmg} pairs`;
  document.getElementById("insp-damage-bar").style.width = `${Math.min(100, (dmg / 20) * 100)}%`;

  document.getElementById("insp-access").textContent = node.access_count ?? 0;
  document.getElementById("insp-age").textContent = `${Math.round(node.age_days ?? 0)}d`;

  let deg = 0;
  if (graphData && graphData.edges) {
    deg = graphData.edges.filter((e) => e.source === node.id || e.target === node.id).length;
  }
  document.getElementById("insp-degree").textContent = deg;
}

document.getElementById("btn-close-inspector")?.addEventListener("click", () => {
  if (inspector) inspector.style.display = "none";
  selectedNode = null;
  renderGraph();
});

// Canvas Interactions
const mainCanvas = document.getElementById("graph-canvas");
if (mainCanvas) {
  mainCanvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    dragStartX = e.clientX - panX;
    dragStartY = e.clientY - panY;
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  mainCanvas.addEventListener("mousemove", (e) => {
    const rect = mainCanvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDragging) {
      panX = e.clientX - dragStartX;
      panY = e.clientY - dragStartY;
      renderGraph();
      return;
    }

    if (!graphData || !nodePositions.length) return;
    const w = rect.width;
    const h = rect.height;

    const worldX = (mouseX - (panX + w / 2)) / zoom + w / 2;
    const worldY = (mouseY - (panY + h / 2)) / zoom + h / 2;

    let hit = null;
    for (let i = 0; i < nodePositions.length; i++) {
      const pos = nodePositions[i];
      const dx = worldX - pos.x;
      const dy = worldY - pos.y;
      if (dx * dx + dy * dy <= 100) {
        hit = graphData.nodes[i];
        break;
      }
    }

    if (hit !== hoveredNode) {
      hoveredNode = hit;
      renderGraph();

      if (hit && tooltip) {
        tooltip.style.display = "block";
        tooltip.style.left = `${mouseX}px`;
        tooltip.style.top = `${mouseY}px`;
        tooltip.innerHTML = `
          <strong>#${hit.id}</strong> ${hit.name || ""}<br>
          Size: ${hit.size_mb} MB | Cluster: ${hit.category}<br>
          ${hit.is_bridge ? '<span style="color:#8b5cf6;">[BRIDGE]</span> ' : ""}
          ${hit.damage ? `<span style="color:#f43f5e;">Damage: ${hit.damage} pairs</span>` : ""}
        `;
      } else if (tooltip) {
        tooltip.style.display = "none";
      }
    }
  });

  mainCanvas.addEventListener("click", () => {
    if (hoveredNode) updateInspector(hoveredNode);
  });

  mainCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.88;
    zoom = Math.max(0.3, Math.min(5.0, zoom * factor));
    renderGraph();
  });
}

document.getElementById("btn-zoom-in")?.addEventListener("click", () => {
  zoom = Math.min(5.0, zoom * 1.25);
  renderGraph();
});
document.getElementById("btn-zoom-out")?.addEventListener("click", () => {
  zoom = Math.max(0.3, zoom * 0.8);
  renderGraph();
});
document.getElementById("btn-reset-view")?.addEventListener("click", () => {
  zoom = 1.0;
  panX = 0;
  panY = 0;
  renderGraph();
});
document.getElementById("btn-toggle-labels")?.addEventListener("click", (e) => {
  showLabels = !showLabels;
  e.target.classList.toggle("active", showLabels);
  renderGraph();
});

// ──────────────── Generation Handler (Client Engine) ────────────────
const btnGenerate = document.getElementById("btn-generate");
if (btnGenerate) {
  btnGenerate.addEventListener("click", () => {
    showLoading("SYNTHESIZING MEMORY TOPOLOGY…");
    setTimeout(() => {
      try {
        const n = parseInt(document.getElementById("inp-n").value, 10);
        const n_categories = parseInt(document.getElementById("inp-categories").value, 10);
        const target_degree = parseFloat(document.getElementById("inp-degree").value);
        const bridge_fraction = parseFloat(document.getElementById("inp-bridges").value);
        const seed = parseInt(document.getElementById("inp-seed").value, 10);

        // Run in pure JS
        clientStore = generateStore(n, n_categories, bridge_fraction, seed);
        clientGraph = buildGraph(clientStore, target_degree, seed);

        const si = new StructuralIndex(clientGraph);
        si.rebuild();
        clientDamage = si.damage;

        graphData = graphToJson(clientGraph, clientStore, clientDamage);

        // Update Telemetry Grid
        document.getElementById("stat-n").textContent = n;
        document.getElementById("stat-edges").textContent = clientGraph.edgeCount();
        document.getElementById("stat-degree").textContent = ((2 * clientGraph.edgeCount()) / n).toFixed(2);
        document.getElementById("stat-size").textContent = `${clientStore.total_size_mb.toFixed(1)} MB`;
        document.getElementById("stat-bridges").textContent = clientStore.memories.filter((m) => m.is_bridge).length;
        document.getElementById("stat-cats").textContent = n_categories;

        const hudCount = document.getElementById("hud-node-count");
        if (hudCount) hudCount.textContent = `V: ${n} | E: ${clientGraph.edgeCount()}`;

        document.getElementById("graph-placeholder").style.display = "none";
        const canvas = document.getElementById("graph-canvas");
        canvas.style.display = "block";
        resizeCanvas("graph-canvas");

        const rect = canvas.getBoundingClientRect();
        nodePositions = layoutGraph(graphData, rect.width, rect.height);

        panX = 0;
        panY = 0;
        zoom = 1.0;
        renderGraph();

        setStatus("GRAPH ONLINE", "online");
      } catch (err) {
        console.error(err);
        setStatus("GENERATION FAILED", "error");
      } finally {
        hideLoading();
      }
    }, 40);
  });
}

// ──────────────── Bar Charts ────────────────
function renderBarChart(canvasId, labels, values, colors, highlightIdx, maxVal = 1.0) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = rect.width || 600;
  const h = rect.height || 280;

  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const paddingLeft = 50;
  const paddingBottom = 40;
  const paddingTop = 30;
  const chartW = w - paddingLeft - 20;
  const chartH = h - paddingTop - paddingBottom;
  const n = values.length;
  const barWidth = Math.max(14, Math.min(36, (chartW / n) * 0.65));
  const gap = chartW / n;

  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yVal = paddingTop + (chartH * g) / 4;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, yVal);
    ctx.lineTo(w - 20, yVal);
    ctx.stroke();

    ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    const labelVal = ((1.0 - g / 4) * maxVal).toFixed(2);
    ctx.fillText(labelVal, paddingLeft - 8, yVal + 3);
  }

  for (let i = 0; i < n; i++) {
    const x = paddingLeft + i * gap + (gap - barWidth) / 2;
    const barH = (Math.max(0, values[i]) / maxVal) * chartH;
    const y = paddingTop + chartH - barH;
    const isTop = i === highlightIdx;

    const grad = ctx.createLinearGradient(x, y, x, y + barH);
    const baseColor = colors[i] || "#6366f1";
    grad.addColorStop(0, isTop ? "#06b6d4" : baseColor);
    grad.addColorStop(1, "rgba(15, 23, 42, 0.8)");

    ctx.fillStyle = grad;
    if (isTop) {
      ctx.shadowColor = "#06b6d4";
      ctx.shadowBlur = 12;
    } else {
      ctx.shadowBlur = 0;
    }

    ctx.beginPath();
    const rad = 4;
    ctx.moveTo(x, y + barH);
    ctx.lineTo(x, y + rad);
    ctx.quadraticCurveTo(x, y, x + rad, y);
    ctx.lineTo(x + barWidth - rad, y);
    ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + rad);
    ctx.lineTo(x + barWidth, y + barH);
    ctx.closePath();
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = isTop ? "#38bdf8" : "rgba(241, 245, 249, 0.9)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(values[i].toFixed(3), x + barWidth / 2, y - 6);

    ctx.fillStyle = isTop ? "#fff" : "rgba(148, 163, 184, 0.7)";
    ctx.font = "9px 'Plus Jakarta Sans', sans-serif";
    ctx.fillText(labels[i], x + barWidth / 2, paddingTop + chartH + 18);
  }
}

// ──────────────── Benchmark Handler (Client Engine) ────────────────
const btnRunCompare = document.getElementById("btn-run-compare");
if (btnRunCompare) {
  btnRunCompare.addEventListener("click", () => {
    if (!clientStore || !clientGraph) {
      alert("Generate a memory graph in the Graph Cockpit first!");
      return;
    }

    showLoading("EXECUTING EVICTION POLICIES BENCHMARK…");
    setTimeout(() => {
      try {
        const budget_pct = parseFloat(document.getElementById("inp-budget").value);
        const selected = Array.from(
          document.querySelectorAll("#policy-checkboxes input:checked")
        ).map((cb) => cb.value);

        const { results } = runPoliciesClient(clientStore, clientGraph, budget_pct, selected);
        if (!results || !results.length) return;

        document.getElementById("compare-placeholder").style.display = "none";
        document.getElementById("compare-results").style.display = "flex";

        let topIdx = 0;
        let topRetention = -1;
        let lruRetention = 0;

        results.forEach((r, idx) => {
          const ret = r.context_retention ?? 0;
          if (ret > topRetention) {
            topRetention = ret;
            topIdx = idx;
          }
          if (r.policy === "lru") lruRetention = ret;
        });

        const topPolicy = results[topIdx];
        document.getElementById("kpi-top-policy").textContent =
          POLICY_LABELS[topPolicy.policy] || topPolicy.policy;
        document.getElementById("kpi-top-desc").textContent = `${(
          topRetention * 100
        ).toFixed(1)}% pairwise context preservation`;

        const diff = lruRetention > 0 ? ((topRetention - lruRetention) / lruRetention) * 100 : 0;
        document.getElementById("kpi-gain").textContent = `+${diff.toFixed(1)}%`;

        const bridgeSurv =
          topPolicy.bridges_survived !== undefined
            ? `${topPolicy.bridges_survived}/${topPolicy.bridges_total || topPolicy.bridges_survived}`
            : "100%";
        document.getElementById("kpi-bridges-saved").textContent = bridgeSurv;

        const labels = results.map((r) => POLICY_LABELS[r.policy] || r.policy);
        const retentions = results.map((r) => r.context_retention ?? 0);
        const importances = results.map((r) => r.importance_retention ?? 0);
        const colors = results.map((r) => POLICY_COLORS[r.policy] || "#6366f1");

        renderBarChart("chart-retention", labels, retentions, colors, topIdx, 1.0);
        renderBarChart("chart-importance", labels, importances, colors, topIdx, 1.0);

        const thead = document.getElementById("results-thead");
        const tbody = document.getElementById("results-tbody");
        thead.innerHTML = `
          <th>POLICY</th>
          <th>CONTEXT RETENTION</th>
          <th>IMPORTANCE RETENTION</th>
          <th>FREED (MB)</th>
          <th>BRIDGES SAVED</th>
          <th>RUNTIME (MS)</th>
        `;
        tbody.innerHTML = "";

        results.forEach((r, idx) => {
          const tr = document.createElement("tr");
          const isWinner = idx === topIdx;
          const color = POLICY_COLORS[r.policy] || "#64748b";
          const cRet = r.context_retention ?? 0;
          const delta = lruRetention > 0 ? cRet - lruRetention : 0;
          const deltaBadge =
            delta !== 0
              ? `<span class="diff-badge ${delta > 0 ? "pos" : "neg"}">${
                  delta > 0 ? "+" : ""
                }${(delta * 100).toFixed(1)}%</span>`
              : "";

          tr.innerHTML = `
            <td>
              <div class="policy-cell">
                <span class="policy-color-dot" style="background:${color}"></span>
                <span>${POLICY_LABELS[r.policy] || r.policy}</span>
                ${isWinner ? '<span class="diff-badge pos">TOP</span>' : ""}
              </div>
            </td>
            <td>
              <span class="metric-pill ${isWinner ? "top" : ""}">${cRet.toFixed(3)}</span>
              ${deltaBadge}
            </td>
            <td><span class="mono">${(r.importance_retention ?? 0).toFixed(3)}</span></td>
            <td><span class="mono">${(r.freed_mb ?? 0).toFixed(1)} MB</span></td>
            <td><span class="mono">${r.bridges_survived ?? "—"}</span></td>
            <td><span class="mono">${r.runtime_ms ?? 0} ms</span></td>
          `;
          tbody.appendChild(tr);
        });

        setStatus("BENCHMARK COMPLETE", "online");
      } catch (err) {
        console.error(err);
        setStatus("BENCHMARK ERROR", "error");
      } finally {
        hideLoading();
      }
    }, 40);
  });
}

// ──────────────── Stepper View (Client Engine) ────────────────
function renderStepperGraph() {
  const canvas = document.getElementById("stepper-canvas");
  if (!canvas || !stepperGraphData || !stepperNodePositions.length) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  ctx.save();
  ctx.translate(stepperPanX + w / 2, stepperPanY + h / 2);
  ctx.scale(stepperZoom, stepperZoom);
  ctx.translate(-w / 2, -h / 2);

  const nodes = stepperGraphData.nodes;
  const edges = stepperGraphData.edges;

  for (const e of edges) {
    const u = e.source;
    const v = e.target;
    if (!stepperNodePositions[u] || !stepperNodePositions[v]) continue;

    const alive = nodes[u]?.alive && nodes[v]?.alive;
    ctx.beginPath();
    ctx.moveTo(stepperNodePositions[u].x, stepperNodePositions[u].y);
    ctx.lineTo(stepperNodePositions[v].x, stepperNodePositions[v].y);

    if (alive) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.lineWidth = 1.0;
    } else {
      ctx.strokeStyle = "rgba(51, 65, 85, 0.2)";
      ctx.lineWidth = 0.6;
    }
    ctx.stroke();
  }

  for (let i = 0; i < nodes.length; i++) {
    const nd = nodes[i];
    const pos = stepperNodePositions[i];
    if (!pos) continue;

    const isJustEvicted = justEvictedId === nd.id;
    const radius = isJustEvicted ? 9 : nd.is_bridge ? 6.5 : 5;

    if (isJustEvicted) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius + 10, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245, 158, 11, 0.3)";
      ctx.fill();
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if (nd.alive) {
      if (nd.damage && nd.damage > 0) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(244, 63, 94, 0.7)";
        ctx.stroke();
      } else if (nd.is_bridge) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(139, 92, 246, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "#8b5cf6";
        ctx.stroke();
      }
    }

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);

    if (isJustEvicted) {
      ctx.fillStyle = "#f59e0b";
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
    } else if (!nd.alive) {
      ctx.fillStyle = "#1e293b";
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1;
    } else {
      ctx.fillStyle = CAT_COLORS[(nd.category || 0) % CAT_COLORS.length];
      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = 1.2;
    }
    ctx.fill();
    ctx.stroke();
  }

  ctx.restore();
}

const btnStepInit = document.getElementById("btn-step-init");
const btnStepNext = document.getElementById("btn-step-next");
const btnStepAuto = document.getElementById("btn-step-auto");
const stepLog = document.getElementById("step-log");
const stepBar = document.getElementById("step-bar");
const stepText = document.getElementById("step-text");
const gaugePct = document.getElementById("gauge-pct");
const stepperBadge = document.getElementById("stepper-status-badge");

function logStep(text, type = "normal") {
  if (!stepLog) return;
  stepEventCount++;
  document.getElementById("log-count").textContent = `${stepEventCount} EVENTS`;

  const el = document.createElement("div");
  el.className = `log-entry ${type}`;
  el.textContent = text;
  stepLog.appendChild(el);
  stepLog.scrollTop = stepLog.scrollHeight;
}

if (btnStepInit) {
  btnStepInit.addEventListener("click", () => {
    if (!clientStore || !clientGraph) {
      alert("Generate a memory graph in the Graph Cockpit first!");
      return;
    }

    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
    }

    const budget_pct = parseFloat(document.getElementById("inp-step-budget").value);
    stepperRequired = clientStore.total_size_mb * (budget_pct / 100.0);
    stepperFreed = 0;
    stepperGraph = clientGraph.clone();

    stepperSI = new StructuralIndex(stepperGraph);
    stepperSI.rebuild();

    const n = clientStore.n;
    stepperStatic = staticFeatures(clientStore);

    const wdeg = new Float64Array(n);
    for (let i = 0; i < n; i++) wdeg[i] = stepperGraph.weightedDegree(i);
    const maxWdeg = Math.max(1.0, ...wdeg);
    const damageScale = Math.max(1.0, ...stepperSI.damage);

    const ctx = structuralScore(stepperSI.damage, wdeg, damageScale, maxWdeg, 0.7);
    stepperVal = retentionValue(stepperStatic, ctx, 0.30);
    stepperRho = stepperVal.map((v, i) => v / clientStore.memories[i].size_mb);

    stepperGraphData = graphToJson(stepperGraph, clientStore, stepperSI.damage);

    document.getElementById("stepper-placeholder").style.display = "none";
    const canvas = document.getElementById("stepper-canvas");
    canvas.style.display = "block";
    resizeCanvas("stepper-canvas");

    const rect = canvas.getBoundingClientRect();
    stepperNodePositions = layoutGraph(stepperGraphData, rect.width, rect.height);
    stepperPanX = 0;
    stepperPanY = 0;
    stepperZoom = 1.0;
    justEvictedId = null;
    renderStepperGraph();

    btnStepNext.disabled = false;
    btnStepAuto.disabled = false;
    stepperBadge.textContent = "INITIALIZED";
    gaugePct.textContent = "0.0%";
    stepBar.style.width = "0%";
    stepText.textContent = `0.00 / ${stepperRequired.toFixed(2)} MB`;

    stepLog.innerHTML = "";
    stepEventCount = 0;
    logStep(`[INIT] Target: Free ${stepperRequired.toFixed(2)} MB (${budget_pct}%)`, "system");
    setStatus("STEPPER READY", "online");
  });
}

function doStepNext() {
  if (!stepperGraph || stepperFreed >= stepperRequired) {
    logStep(`[COMPLETE] Freed ${stepperFreed.toFixed(2)} MB`, "complete");
    stepperBadge.textContent = "COMPLETE";
    btnStepNext.disabled = true;
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
    }
    return false;
  }

  const n = clientStore.n;
  let bestV = -1, bestRho = Infinity;
  for (let i = 0; i < n; i++) {
    if (stepperGraph.alive[i] && stepperRho[i] < bestRho) {
      bestRho = stepperRho[i];
      bestV = i;
    }
  }

  if (bestV < 0) {
    stepperBadge.textContent = "COMPLETE";
    btnStepNext.disabled = true;
    return false;
  }

  const neigh = stepperGraph.neighbours(bestV);
  const wasCut = stepperSI.damage[bestV] > 0;
  const isBridge = clientStore.memories[bestV].is_bridge;

  stepperGraph.delete(bestV);
  justEvictedId = bestV;
  stepperFreed += clientStore.memories[bestV].size_mb;

  // Bridge repair
  if (neigh.length > 1) {
    for (let a = 0; a < neigh.length; a++) {
      for (let b = a + 1; b < neigh.length; b++) {
        const u = neigh[a], v = neigh[b];
        if (!stepperGraph.adj[u].has(v)) {
          const s = clientStore.similarity(u, v);
          if (s >= 0.25) stepperGraph.addEdge(u, v, s);
        }
      }
    }
  }

  // Local Tarjan recompute
  const changed = stepperSI.updateAfterDelete(bestV);
  const wdeg = new Float64Array(n);
  for (let i = 0; i < n; i++) if (stepperGraph.alive[i]) wdeg[i] = stepperGraph.weightedDegree(i);
  const maxWdeg = Math.max(1.0, ...wdeg);
  const damageScale = Math.max(1.0, ...stepperSI.damage);

  if (changed.length) {
    const newCtx = structuralScore(stepperSI.damage, wdeg, damageScale, maxWdeg, 0.7);
    const newVal = retentionValue(stepperStatic, newCtx, 0.30);
    for (const u of changed) {
      stepperVal[u] = newVal[u];
      stepperRho[u] = newVal[u] / clientStore.memories[u].size_mb;
    }
  }

  stepperGraphData = graphToJson(stepperGraph, clientStore, stepperSI.damage);
  renderStepperGraph();

  const pct = Math.min(100, (stepperFreed / stepperRequired) * 100);
  gaugePct.textContent = `${pct.toFixed(1)}%`;
  stepBar.style.width = `${pct}%`;
  stepText.textContent = `${stepperFreed.toFixed(2)} / ${stepperRequired.toFixed(2)} MB`;

  const m = clientStore.memories[bestV];
  let msg = `[EVICT #${bestV}] ${m.size_mb} MB | ρ=${bestRho.toFixed(4)}`;
  if (isBridge) msg += " | [CRITICAL BRIDGE]";
  if (wasCut) msg += " | [CUT VERTEX]";
  logStep(msg, isBridge ? "bridge-alert" : "evict");

  return true;
}

if (btnStepNext) {
  btnStepNext.addEventListener("click", () => doStepNext());
}

if (btnStepAuto) {
  btnStepAuto.addEventListener("click", () => {
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = null;
      btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
      btnStepAuto.classList.remove("active");
    } else {
      btnStepAuto.querySelector(".t-txt").textContent = "PAUSE";
      btnStepAuto.classList.add("active");
      autoPlayTimer = setInterval(() => {
        const ok = doStepNext();
        if (!ok && autoPlayTimer) {
          clearInterval(autoPlayTimer);
          autoPlayTimer = null;
          btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
          btnStepAuto.classList.remove("active");
        }
      }, autoPlaySpeed);
    }
  });
}

document.querySelectorAll(".speed-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".speed-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    autoPlaySpeed = parseInt(pill.dataset.speed, 10);
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = setInterval(() => {
        const ok = doStepNext();
        if (!ok && autoPlayTimer) {
          clearInterval(autoPlayTimer);
          autoPlayTimer = null;
          btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
        }
      }, autoPlaySpeed);
    }
  });
});

const stepperCanvas = document.getElementById("stepper-canvas");
if (stepperCanvas) {
  stepperCanvas.addEventListener("mousedown", (e) => {
    isStepperDragging = true;
    stepperDragStartX = e.clientX - stepperPanX;
    stepperDragStartY = e.clientY - stepperPanY;
  });

  window.addEventListener("mouseup", () => {
    isStepperDragging = false;
  });

  stepperCanvas.addEventListener("mousemove", (e) => {
    if (isStepperDragging) {
      stepperPanX = e.clientX - stepperDragStartX;
      stepperPanY = e.clientY - stepperDragStartY;
      renderStepperGraph();
    }
  });

  stepperCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.88;
    stepperZoom = Math.max(0.3, Math.min(5.0, stepperZoom * factor));
    renderStepperGraph();
  });
}

document.getElementById("btn-stepper-zoom-in")?.addEventListener("click", () => {
  stepperZoom = Math.min(5.0, stepperZoom * 1.25);
  renderStepperGraph();
});
document.getElementById("btn-stepper-zoom-out")?.addEventListener("click", () => {
  stepperZoom = Math.max(0.3, stepperZoom * 0.8);
  renderStepperGraph();
});
document.getElementById("btn-stepper-reset")?.addEventListener("click", () => {
  stepperZoom = 1.0;
  stepperPanX = 0;
  stepperPanY = 0;
  renderStepperGraph();
});
