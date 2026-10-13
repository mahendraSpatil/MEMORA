/* ═══════════════════════════════════════════════════════════════
   MEMORA UI — Next-Gen Interactive Graph Engine & Controls
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

// ──────────────── State ────────────────
let graphData = null;
let nodePositions = [];
let hoveredNode = null;
let selectedNode = null;
let showLabels = true;

// Pan & Zoom state for main canvas
let panX = 0;
let panY = 0;
let zoom = 1.0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

// Stepper canvas Pan & Zoom
let stepperPanX = 0;
let stepperPanY = 0;
let stepperZoom = 1.0;
let isStepperDragging = false;
let stepperDragStartX = 0;
let stepperDragStartY = 0;

// Stepper animation state
let autoPlayTimer = null;
let autoPlaySpeed = 400; // ms
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

    // Resize canvas if needed
    if (btn.dataset.tab === "playground") {
      resizeCanvas("graph-canvas");
      renderGraph();
    } else if (btn.dataset.tab === "stepper") {
      resizeCanvas("stepper-canvas");
      renderStepperGraph();
    }
  });
});

// ──────────────── Sliders & Controls ────────────────
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

const btnRandomSeed = document.getElementById("btn-random-seed");
if (btnRandomSeed) {
  btnRandomSeed.addEventListener("click", () => {
    document.getElementById("inp-seed").value = Math.floor(Math.random() * 10000);
  });
}

// ──────────────── Policy Checkbox Chips ────────────────
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

// ──────────────── API Helpers ────────────────
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

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// ──────────────── High-DPI Canvas Scaling ────────────────
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

// ──────────────── Force Layout Simulation ────────────────
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

    // Node-node repulsion
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

    // Edge spring tension
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

    // Center gravitational pull
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

// ──────────────── Main Graph Rendering ────────────────
function renderGraph() {
  const canvas = document.getElementById("graph-canvas");
  if (!canvas || !graphData || !nodePositions.length) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);

  // Apply Pan & Zoom
  ctx.save();
  ctx.translate(panX + w / 2, panY + h / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-w / 2, -h / 2);

  // Draw Edges
  const nodes = graphData.nodes;
  const edges = graphData.edges;

  for (const e of edges) {
    const u = e.source;
    const v = e.target;
    if (!nodePositions[u] || !nodePositions[v]) continue;

    const uAlive = nodes[u] ? nodes[u].alive : true;
    const vAlive = nodes[v] ? nodes[v].alive : true;
    const alive = uAlive && vAlive;

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

    // Outer Aura for Articulation Points & Bridges
    if (nd.alive) {
      if (nd.damage && nd.damage > 0) {
        // Articulation point glow
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 5, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(244, 63, 94, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(244, 63, 94, 0.7)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      } else if (nd.is_bridge) {
        // Bridge node halo
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(139, 92, 246, 0.25)";
        ctx.fill();
        ctx.strokeStyle = "rgba(139, 92, 246, 0.8)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // Node Body
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

    // Node Labels
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

const btnCloseInsp = document.getElementById("btn-close-inspector");
if (btnCloseInsp) {
  btnCloseInsp.addEventListener("click", () => {
    inspector.style.display = "none";
    selectedNode = null;
    renderGraph();
  });
}

// ──────────────── Canvas Mouse Interaction ────────────────
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

    // Hit-testing nodes
    if (!graphData || !nodePositions.length) return;
    const w = rect.width;
    const h = rect.height;

    // Transform screen mouse coords to canvas world coords
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
    if (hoveredNode) {
      updateInspector(hoveredNode);
    }
  });

  mainCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.88;
    zoom = Math.max(0.3, Math.min(5.0, zoom * factor));
    renderGraph();
  });
}

// Canvas floating zoom buttons
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

// ──────────────── API Generate Graph ────────────────
const btnGenerate = document.getElementById("btn-generate");
if (btnGenerate) {
  btnGenerate.addEventListener("click", async () => {
    showLoading("SYNTHESIZING MEMORY TOPOLOGY…");
    try {
      const n = parseInt(document.getElementById("inp-n").value, 10);
      const n_categories = parseInt(document.getElementById("inp-categories").value, 10);
      const target_degree = parseFloat(document.getElementById("inp-degree").value);
      const bridge_fraction = parseFloat(document.getElementById("inp-bridges").value);
      const seed = parseInt(document.getElementById("inp-seed").value, 10);

      const data = await apiPost("/api/generate", {
        n,
        n_categories,
        target_degree,
        bridge_fraction,
        seed,
      });

      graphData = data.graph;

      // Update Telemetry Grid
      document.getElementById("stat-n").textContent = data.n;
      document.getElementById("stat-edges").textContent = data.edge_count;
      document.getElementById("stat-degree").textContent = data.avg_degree;
      document.getElementById("stat-size").textContent = `${data.total_size_mb} MB`;
      document.getElementById("stat-bridges").textContent = data.bridges;
      document.getElementById("stat-cats").textContent = data.categories;

      const hudCount = document.getElementById("hud-node-count");
      if (hudCount) hudCount.textContent = `V: ${data.n} | E: ${data.edge_count}`;

      // Switch Canvas Viewports
      document.getElementById("graph-placeholder").style.display = "none";
      const canvas = document.getElementById("graph-canvas");
      canvas.style.display = "block";
      resizeCanvas("graph-canvas");

      // Layout Graph Physics
      const rect = canvas.getBoundingClientRect();
      nodePositions = layoutGraph(graphData, rect.width, rect.height);

      // Reset Pan & Zoom
      panX = 0;
      panY = 0;
      zoom = 1.0;
      renderGraph();

      // Enable stepper ready state
      setStatus("GRAPH ONLINE", "online");
    } catch (err) {
      console.error(err);
      setStatus("GENERATION FAILED", "error");
    } finally {
      hideLoading();
    }
  });
}

// ──────────────── Benchmark Execution & Charts ────────────────
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

  // Background Gridlines
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

  // Draw Bars
  for (let i = 0; i < n; i++) {
    const x = paddingLeft + i * gap + (gap - barWidth) / 2;
    const barH = (Math.max(0, values[i]) / maxVal) * chartH;
    const y = paddingTop + chartH - barH;
    const isTop = i === highlightIdx;

    // Glowing Neon Gradient
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

    // Rounded Top Bar
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

    // Value Label on Top
    ctx.fillStyle = isTop ? "#38bdf8" : "rgba(241, 245, 249, 0.9)";
    ctx.font = "10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(values[i].toFixed(3), x + barWidth / 2, y - 6);

    // X Axis Label
    ctx.fillStyle = isTop ? "#fff" : "rgba(148, 163, 184, 0.7)";
    ctx.font = "9px 'Plus Jakarta Sans', sans-serif";
    ctx.fillText(labels[i], x + barWidth / 2, paddingTop + chartH + 18);
  }
}

const btnRunCompare = document.getElementById("btn-run-compare");
if (btnRunCompare) {
  btnRunCompare.addEventListener("click", async () => {
    if (!graphData) {
      alert("Generate a graph in the Graph Cockpit first!");
      return;
    }

    showLoading("EXECUTING EVICTION POLICIES BENCHMARK…");
    try {
      const budget_pct = parseFloat(document.getElementById("inp-budget").value);
      const selected = Array.from(
        document.querySelectorAll("#policy-checkboxes input:checked")
      ).map((cb) => cb.value);

      const res = await apiPost("/api/run", {
        budget_pct,
        policies: selected,
      });

      const results = res.results;
      if (!results || !results.length) return;

      document.getElementById("compare-placeholder").style.display = "none";
      document.getElementById("compare-results").style.display = "flex";

      // Find top policy by context retention
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

      // Populate Charts
      const labels = results.map((r) => POLICY_LABELS[r.policy] || r.policy);
      const retentions = results.map((r) => r.context_retention ?? 0);
      const importances = results.map((r) => r.importance_retention ?? 0);
      const colors = results.map((r) => POLICY_COLORS[r.policy] || "#6366f1");

      renderBarChart("chart-retention", labels, retentions, colors, topIdx, 1.0);
      renderBarChart("chart-importance", labels, importances, colors, topIdx, 1.0);

      // Populate Matrix Table
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
  });
}

// ──────────────── Stepper Eviction Simulation ────────────────
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

  // Draw Edges
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

  // Draw Nodes
  for (let i = 0; i < nodes.length; i++) {
    const nd = nodes[i];
    const pos = stepperNodePositions[i];
    if (!pos) continue;

    const isJustEvicted = justEvictedId === nd.id;
    const radius = isJustEvicted ? 9 : nd.is_bridge ? 6.5 : 5;

    // Shockwave pulse for just evicted
    if (isJustEvicted) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, radius + 10, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245, 158, 11, 0.3)";
      ctx.fill();
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Halo
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

// Stepper controls
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
  btnStepInit.addEventListener("click", async () => {
    if (!graphData) {
      alert("Generate a memory graph in the Graph Cockpit first!");
      return;
    }

    showLoading("INITIALIZING CAFA STEPPER…");
    try {
      if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
        btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
      }

      const budget_pct = parseFloat(document.getElementById("inp-step-budget").value);
      const res = await apiPost("/api/step", { action: "init", budget_pct });

      stepperGraphData = res.graph;
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
      stepText.textContent = `0.00 / ${res.required_mb} MB`;

      stepLog.innerHTML = "";
      stepEventCount = 0;
      logStep(`[INIT] Target: Free ${res.required_mb} MB (${budget_pct}%)`, "system");
      setStatus("STEPPER READY", "online");
    } catch (err) {
      console.error(err);
    } finally {
      hideLoading();
    }
  });
}

async function doStepNext() {
  try {
    const res = await apiPost("/api/step", { action: "next" });

    if (res.status === "complete") {
      logStep(`[COMPLETE] Freed ${res.freed} MB (${res.n_deleted} evicted)`, "complete");
      stepperBadge.textContent = "COMPLETE";
      btnStepNext.disabled = true;
      if (autoPlayTimer) {
        clearInterval(autoPlayTimer);
        autoPlayTimer = null;
        btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
      }
      return false;
    }

    stepperGraphData = res.graph;
    justEvictedId = res.evicted.id;
    renderStepperGraph();

    const pct = Math.min(100, res.pct_done);
    gaugePct.textContent = `${pct.toFixed(1)}%`;
    stepBar.style.width = `${pct}%`;
    stepText.textContent = `${res.freed} / ${res.required_mb} MB`;

    const ev = res.evicted;
    let msg = `[EVICT #${ev.id}] ${ev.size_mb} MB | ρ=${ev.rho.toFixed(4)}`;
    if (ev.is_bridge) msg += " | [CRITICAL BRIDGE]";
    if (ev.was_articulation) msg += " | [CUT VERTEX]";
    logStep(msg, ev.is_bridge ? "bridge-alert" : "evict");

    return true;
  } catch (err) {
    console.error(err);
    return false;
  }
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
      autoPlayTimer = setInterval(async () => {
        const ok = await doStepNext();
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

// Speed toggles
document.querySelectorAll(".speed-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    document.querySelectorAll(".speed-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    autoPlaySpeed = parseInt(pill.dataset.speed, 10);
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = setInterval(async () => {
        const ok = await doStepNext();
        if (!ok && autoPlayTimer) {
          clearInterval(autoPlayTimer);
          autoPlayTimer = null;
          btnStepAuto.querySelector(".t-txt").textContent = "AUTO";
        }
      }, autoPlaySpeed);
    }
  });
});

// Stepper Canvas Pan & Zoom
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
