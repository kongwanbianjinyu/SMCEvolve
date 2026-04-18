"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const ISLAND_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];
const MODEL_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"];

let CURRENT = null;   // {runId, events, islands, rmin, rmax}

window.addEventListener("DOMContentLoaded", main);

async function main() {
  const runs = await fetchJSON("/api/runs");
  const list = document.getElementById("runs");
  list.innerHTML = "";
  if (runs.length === 0) {
    list.innerHTML = "<li style='color:#8b95a3'>(no runs yet)</li>";
    return;
  }
  for (const r of runs) {
    const li = document.createElement("li");
    li.textContent = r.id;
    li.title = `${r.id}\n${(r.size_bytes / 1024).toFixed(1)} KB`;
    li.dataset.runId = r.id;
    li.addEventListener("click", () => loadRun(r.id));
    list.appendChild(li);
  }
  loadRun(runs[0].id);

  document.getElementById("detail-close").addEventListener("click", hideDetail);

  // Tab switching
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".view").forEach(v => {
        v.classList.toggle("hidden", v.id !== `view-${target}`);
      });
    });
  });
}

async function loadRun(runId) {
  document.querySelectorAll("#runs li").forEach(li => {
    li.classList.toggle("selected", li.dataset.runId === runId);
  });
  const data = await fetchJSON("/api/run/" + encodeURIComponent(runId));
  const islands = groupByIsland(data.events);
  const [rmin, rmax] = computeRewardRange(data.events);
  CURRENT = { runId, events: data.events, islands, rmin, rmax };
  renderHeader(runId, islands, rmin, rmax);
  renderIslands(islands, rmin, rmax);
  renderCurve(islands);
  renderCost(data.events, islands);
  renderKernel(data.events, islands);
  hideDetail();
}

function renderHeader(runId, islands, rmin, rmax) {
  const h = document.getElementById("run-header");
  const totalIters = islands.reduce((s, isl) => s + isl.iterations.length, 0);
  h.innerHTML = `
    <div class="run-name">${runId}</div>
    <div>${islands.length} island(s) · ${totalIters} total iterations · reward range [${rmin.toFixed(4)}, ${rmax.toFixed(4)}]</div>
  `;
}

// ----- data shaping --------------------------------------------------------

function groupByIsland(events) {
  const byIsland = new Map();
  for (const e of events) {
    if (e.island === undefined) continue;
    if (!byIsland.has(e.island)) {
      byIsland.set(e.island, { id: e.island, init: null, iters: new Map() });
    }
    const isl = byIsland.get(e.island);
    if (e.type === "init") {
      isl.init = e;
    } else if (e.type === "resample") {
      const it = ensureIter(isl, e.iteration);
      it.resample = e;
    } else if (e.type === "step_summary") {
      const it = ensureIter(isl, e.iteration);
      it.summary = e;
    } else if (e.type === "proposal" || e.type === "mh_step") {
      const it = ensureIter(isl, e.iteration);
      const p = e.particle_idx;
      if (!it.proposals[p]) it.proposals[p] = [];
      it.proposals[p].push(e);
    }
  }
  const islands = [...byIsland.values()].sort((a, b) => a.id - b.id);
  for (const isl of islands) {
    isl.iterations = [...isl.iters.values()]
      .filter(it => it.resample)
      .sort((a, b) => a.iter - b.iter);
    delete isl.iters;
  }
  return islands;
}

function ensureIter(isl, iter) {
  if (!isl.iters.has(iter)) {
    isl.iters.set(iter, { iter, resample: null, summary: null, proposals: {} });
  }
  return isl.iters.get(iter);
}

function computeRewardRange(events) {
  let rmin = Infinity;
  let rmax = -Infinity;
  for (const e of events) {
    const candidates = [];
    if (e.type === "proposal" || e.type === "mh_step") {
      candidates.push(e.parent_reward, e.child_reward);
    } else if (e.type === "step_summary") {
      if (Array.isArray(e.rewards)) candidates.push(...e.rewards);
    } else if (e.type === "resample") {
      if (Array.isArray(e.parents)) candidates.push(...e.parents.map(p => p.reward));
    }
    for (const v of candidates) {
      if (typeof v === "number" && isFinite(v)) {
        if (v < rmin) rmin = v;
        if (v > rmax) rmax = v;
      }
    }
  }
  if (!isFinite(rmin)) rmin = 0;
  if (!isFinite(rmax)) rmax = 1;
  if (rmin === rmax) rmax = rmin + 1;
  return [rmin, rmax];
}

// ----- rendering ----------------------------------------------------------

function renderIslands(islands, rmin, rmax) {
  const root = document.getElementById("islands");
  root.innerHTML = "";
  if (islands.some(isl => isl.iterations.length > 0)) {
    const btn = document.createElement("button");
    btn.className = "export-all-flow-btn";
    btn.textContent = "\u2193 Export all flow diagrams";
    btn.title = "Download all flow diagrams combined as one SVG";
    btn.addEventListener("click", downloadAllFlowSVG);
    root.appendChild(btn);
  }
  for (const isl of islands) {
    root.appendChild(renderIsland(isl, rmin, rmax));
  }
}

function renderIsland(isl, rmin, rmax) {
  const div = document.createElement("div");
  div.className = "island";
  const initText = isl.init && Array.isArray(isl.init.initial_rewards)
    ? `init rewards: [${isl.init.initial_rewards.map(r => r.toFixed(2)).join(", ")}]`
    : "";
  div.innerHTML = `<h2>Island ${isl.id} <span style="font-weight:400; color:#8b95a3; font-size:11px;">${initText}</span></h2>`;
  for (const it of isl.iterations) {
    div.appendChild(renderIteration(isl.id, it, rmin, rmax));
  }
  return div;
}

function renderIteration(islandId, it, rmin, rmax) {
  const r = it.resample;
  const N = r.parents.length;

  const wrap = document.createElement("div");
  wrap.className = "iter";

  const stats = document.createElement("div");
  stats.className = "stats";
  const summary = it.summary;
  stats.innerHTML = `
    <div class="iter-num">iter ${it.iter}</div>
    <div><span class="stat-key">λ</span> ${r.lambda.toFixed(4)}</div>
    <div><span class="stat-key">Δβ</span> ${r.delta_beta.toFixed(4)}</div>
    <div><span class="stat-key">β_t</span> ${r.beta_t.toFixed(4)}</div>
    <div><span class="stat-key">ESS</span> ${r.ess_at_lambda.toFixed(2)} / ${N}</div>
    ${summary ? `<div><span class="stat-key">best</span> ${summary.best_reward.toFixed(4)}</div>` : ""}
    ${summary ? `<div><span class="stat-key">mean</span> ${summary.mean_reward.toFixed(4)}</div>` : ""}
  `;
  wrap.appendChild(stats);

  // SVG dimensions
  const padX = 80;
  const dx = Math.max(34, Math.min(60, 480 / Math.max(N, 1)));
  const radius = Math.min(13, dx / 2.6);
  const yReweight = 30;
  const yResample = 110;
  const yMutate = 200;
  const propDy = Math.max(radius * 2 + 6, 28);
  let maxPropRows = 0;
  for (let i = 0; i < N; i++) {
    maxPropRows = Math.max(maxPropRows, (it.proposals[i] || []).length);
  }
  const W = padX + (N - 1) * dx + 30;
  const H = yMutate + Math.max(0, maxPropRows - 1) * propDy + radius + 20;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.classList.add("flow");

  // arrow marker
  const defs = document.createElementNS(SVG_NS, "defs");
  defs.innerHTML = `
    <marker id="arrowhead-${islandId}-${it.iter}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L6,3 z" fill="#9ca3af"/>
    </marker>`;
  svg.appendChild(defs);
  const markerUrl = `url(#arrowhead-${islandId}-${it.iter})`;

  // group labels
  for (const [text, y] of [["reweight", yReweight], ["resample", yResample], ["mutate", yMutate]]) {
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", 6);
    t.setAttribute("y", y + 4);
    t.classList.add("group-label");
    t.textContent = text;
    svg.appendChild(t);
  }

  // weight scaling
  const wmax = Math.max(...r.weights);

  // reweight row (parents); area ∝ weight. Parents that arrived via
  // migration in the previous epoch get a thick ring in the source
  // island's color.
  for (let i = 0; i < N; i++) {
    const p = r.parents[i];
    const w = r.weights[i];
    const rad = Math.max(3, radius * Math.sqrt(w / (wmax || 1)));
    const c = makeCircle(cx(i), yReweight, rad, colorOf(p.reward, rmin, rmax));
    let title = `parent[${i}]\nreward=${fmt(p.reward)}\nweight=${w.toExponential(2)}`;
    if (p.migrated_from !== undefined && p.migrated_from !== null) {
      const srcColor = ISLAND_COLORS[p.migrated_from % ISLAND_COLORS.length];
      c.setAttribute("stroke", srcColor);
      c.setAttribute("stroke-width", 2.5);
      title += `\n⇠ migrated from island ${p.migrated_from}`;
    }
    c.dataset.title = title;
    svg.appendChild(c);
  }

  // resample row + arrows
  for (let i = 0; i < N; i++) {
    const a = r.ancestors[i];
    const parentReward = r.parents[a].reward;
    const c = makeCircle(cx(i), yResample, radius, colorOf(parentReward, rmin, rmax));
    c.dataset.title = `resampled[${i}] ← parent[${a}]\nreward=${fmt(parentReward)}`;
    svg.appendChild(c);
    svg.appendChild(makeArrow(cx(a), yReweight + radius, cx(i), yResample - radius, markerUrl));
  }

  // mutate rows: one circle per proposal (best-of-K)
  for (let i = 0; i < N; i++) {
    const props = it.proposals[i] || [];
    let prevY = yResample + radius;
    for (let k = 0; k < props.length; k++) {
      const m = props[k];
      const yk = yMutate + k * propDy;
      const fillReward = m.accepted ? m.child_reward : m.parent_reward;
      const c = makeCircle(cx(i), yk, radius, colorOf(fillReward, rmin, rmax));
      if (m.skipped) {
        c.setAttribute("stroke-dasharray", "1,2");
        c.setAttribute("fill-opacity", "0.35");
      } else if (!m.accepted) {
        c.setAttribute("stroke-dasharray", "2,2");
        c.setAttribute("fill-opacity", "0.55");
      }
      // Kernel encoding: stroke colour = edit_mode, stroke width = inspiration.
      const kinfo = kernelInfo(m);
      const kvis = kernelVisual(kinfo);
      c.setAttribute("stroke", kvis.stroke);
      c.setAttribute("stroke-width", kvis.strokeWidth);
      const status = m.skipped
        ? `SKIPPED (${m.skipped})`
        : (m.accepted ? "IMPROVED" : "not improved");
      c.dataset.title =
        `p${i} · proposal ${k}\n` +
        `kernel ${kinfo.label}\n` +
        `parent r=${fmt(m.parent_reward)}\n` +
        `child  r=${fmt(m.child_reward)}\n` +
        status;
      c.dataset.island = islandId;
      c.dataset.iteration = it.iter;
      c.dataset.particleIdx = i;
      c.dataset.kStep = k;
      c.classList.add("clickable");
      svg.appendChild(c);
      svg.appendChild(makeLine(cx(i), prevY, cx(i), yk - radius));
      prevY = yk + radius;
    }
  }

  function cx(i) { return padX + i * dx; }

  // tooltip + click
  svg.addEventListener("mousemove", e => {
    if (e.target.dataset && e.target.dataset.title) {
      showTooltip(e, e.target.dataset.title);
    } else {
      hideTooltip();
    }
  });
  svg.addEventListener("mouseleave", hideTooltip);
  svg.addEventListener("click", e => {
    if (e.target.classList && e.target.classList.contains("clickable")) {
      const p = parseInt(e.target.dataset.particleIdx, 10);
      const kIdxRaw = e.target.dataset.kStep;
      const kIdx = kIdxRaw != null ? parseInt(kIdxRaw, 10) : null;
      showDetail(islandId, it.iter, p, kIdx, it);
    }
  });

  wrap.appendChild(wrapSvgWithExport(svg, `flow_island${islandId}_iter${it.iter}.svg`));
  return wrap;
}

// ----- svg helpers --------------------------------------------------------

function makeCircle(cx, cy, r, fill) {
  const c = document.createElementNS(SVG_NS, "circle");
  c.setAttribute("cx", cx);
  c.setAttribute("cy", cy);
  c.setAttribute("r", r);
  c.setAttribute("fill", fill);
  c.setAttribute("stroke", "#374151");
  c.setAttribute("stroke-width", 1);
  return c;
}

function makeArrow(x1, y1, x2, y2, markerUrl) {
  const l = document.createElementNS(SVG_NS, "line");
  l.setAttribute("x1", x1);
  l.setAttribute("y1", y1);
  l.setAttribute("x2", x2);
  l.setAttribute("y2", y2);
  l.setAttribute("stroke", "#9ca3af");
  l.setAttribute("stroke-width", 1);
  l.setAttribute("marker-end", markerUrl);
  return l;
}

function makeLine(x1, y1, x2, y2) {
  const l = document.createElementNS(SVG_NS, "line");
  l.setAttribute("x1", x1);
  l.setAttribute("y1", y1);
  l.setAttribute("x2", x2);
  l.setAttribute("y2", y2);
  l.setAttribute("stroke", "#d1d5db");
  l.setAttribute("stroke-width", 1);
  l.setAttribute("stroke-dasharray", "2,3");
  return l;
}

// ----- detail panel ------------------------------------------------------

function showDetail(islandId, iter, particleIdx, kStepIdx, it) {
  const props = it.proposals[particleIdx] || [];
  const shown =
    kStepIdx != null && props[kStepIdx]
      ? props[kStepIdx]
      : props[props.length - 1];
  const r = it.resample;
  const ancestor = r.ancestors[particleIdx];
  const parentReward = r.parents[ancestor].reward;

  const shownIdx =
    shown != null ? props.indexOf(shown) : -1;
  document.getElementById("detail-title").textContent =
    shownIdx >= 0
      ? `island ${islandId} · iter ${iter} · p${particleIdx} · proposal ${shownIdx}`
      : `island ${islandId} · iter ${iter} · particle ${particleIdx}`;

  const finalReward = props.reduce(
    (acc, m) => (m.accepted ? m.child_reward : acc), parentReward,
  );
  const improved = props.filter(m => m.accepted).length;

  let kernelLine = "";
  if (shown) {
    const k = kernelInfo(shown);
    if (k.name) {
      const parts = [`<b>kernel</b> ${k.name}`];
      if (k.editMode) parts.push(`<span class="kbadge km-${k.editMode}">${k.editMode}</span>`);
      if (k.nInspo > 0) parts.push(`<b>inspo</b> ${k.nInspo}`);
      if (k.fallback) parts.push(`<span class="kbadge km-fallback">fallback</span>`);
      if (k.issues.length) {
        parts.push(`<span class="kbadge km-issues" title="${k.issues.join(', ')}">${k.issues.length} issue${k.issues.length === 1 ? '' : 's'}</span>`);
      }
      kernelLine = `<div class="kernel-line">${parts.join(" ")}</div>`;
    }
  }

  document.getElementById("detail-info").innerHTML = `
    <span><b>parent</b> #${ancestor} (r=${fmt(parentReward)})</span>
    <span><b>final r</b> ${fmt(finalReward)}</span>
    <span><b>improved</b> ${improved}/${props.length}</span>
    <span><b>β_t</b> ${r.beta_t.toFixed(4)}</span>
    ${kernelLine}
  `;

  if (shown) {
    document.getElementById("detail-prompt").textContent = shown.prompt || "(no prompt)";
    document.getElementById("detail-response").textContent = shown.response || "(no response)";
    document.getElementById("detail-program").textContent = shown.program || "(no program)";
  } else {
    document.getElementById("detail-prompt").textContent = "(no proposal recorded)";
    document.getElementById("detail-response").textContent = "";
    document.getElementById("detail-program").textContent = "";
  }
  document.getElementById("detail").classList.remove("hidden");
}

function hideDetail() {
  document.getElementById("detail").classList.add("hidden");
}

// ----- tooltip + helpers -------------------------------------------------

function showTooltip(evt, text) {
  const tip = document.getElementById("tooltip");
  tip.textContent = text;
  tip.classList.remove("hidden");
  const pad = 12;
  tip.style.left = (evt.clientX + pad) + "px";
  tip.style.top = (evt.clientY + pad) + "px";
}

function hideTooltip() {
  document.getElementById("tooltip").classList.add("hidden");
}

// ----- kernel metadata ---------------------------------------------------

const KERNEL_STROKE = {
  diff: "#2563eb",      // blue — local SEARCH/REPLACE edits
  rewrite: "#e67e22",   // orange — full-program rewrite
};
const KERNEL_STROKE_FALLBACK = "#374151"; // old default, for records missing metadata

function kernelInfo(m) {
  const meta = (m && m.proposal_metadata) || {};
  const name = meta.kernel || null;
  const editMode = meta.edit_mode || null;
  const needsInspo = !!meta.needs_inspiration;
  const nInspo = typeof meta.n_inspirations === "number" ? meta.n_inspirations : 0;
  const fallback = !!meta.fallback_from_inspo;
  const issues = Array.isArray(meta.parse_issues) ? meta.parse_issues : [];
  let label;
  if (!name) {
    label = "(unknown)";
  } else {
    const parts = [name];
    if (nInspo > 0) parts.push(`+${nInspo} inspo`);
    if (fallback) parts.push("fallback");
    label = parts.join(" · ");
  }
  return { name, editMode, needsInspo, nInspo, fallback, issues, label };
}

function kernelVisual(kinfo) {
  const stroke = KERNEL_STROKE[kinfo.editMode] || KERNEL_STROKE_FALLBACK;
  const strokeWidth = kinfo.needsInspo ? 3.5 : 1;
  return { stroke, strokeWidth };
}

function colorOf(reward, rmin, rmax) {
  if (typeof reward !== "number" || !isFinite(reward)) return "#9ca3af";
  const t = Math.max(0, Math.min(1, (reward - rmin) / (rmax - rmin || 1)));
  // red(196,56,56) -> yellow(216,167,43) -> green(47,170,84)
  if (t < 0.5) {
    const k = t / 0.5;
    return rgb(lerp(196, 216, k), lerp(56, 167, k), lerp(56, 43, k));
  }
  const k = (t - 0.5) / 0.5;
  return rgb(lerp(216, 47, k), lerp(167, 170, k), lerp(43, 84, k));
}

function lerp(a, b, t) { return a + (b - a) * t; }
function rgb(r, g, b) { return `rgb(${r|0},${g|0},${b|0})`; }
function fmt(x) { return (typeof x === "number" && isFinite(x)) ? x.toFixed(4) : String(x); }

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// ----- score curve view ---------------------------------------------------

function renderCurve(islands) {
  const root = document.getElementById("curve-chart");
  root.innerHTML = "";

  // Build per-island series. iter 0 = init rewards, iter t = step_summary.rewards.
  const series = islands.map((isl, idx) => {
    const points = [];
    const bestEver = [];
    let runningBest = -Infinity;

    if (isl.init && Array.isArray(isl.init.initial_rewards)) {
      for (const r of isl.init.initial_rewards) {
        if (typeof r === "number" && isFinite(r)) {
          points.push({ iter: 0, reward: r });
          if (r > runningBest) runningBest = r;
        }
      }
      if (runningBest > -Infinity) bestEver.push({ iter: 0, best: runningBest });
    }
    for (const it of isl.iterations) {
      if (!it.summary || !Array.isArray(it.summary.rewards)) continue;
      for (const r of it.summary.rewards) {
        if (typeof r === "number" && isFinite(r)) {
          points.push({ iter: it.iter, reward: r });
          if (r > runningBest) runningBest = r;
        }
      }
      if (runningBest > -Infinity) bestEver.push({ iter: it.iter, best: runningBest });
    }

    return {
      id: isl.id,
      color: ISLAND_COLORS[idx % ISLAND_COLORS.length],
      points,
      bestEver,
    };
  });

  const allPoints = series.flatMap(s => s.points);
  if (allPoints.length === 0) {
    root.innerHTML = "<div style='color:#6b7280; font-size:12px;'>No reward data yet.</div>";
    return;
  }

  const xs = allPoints.map(p => p.iter);
  const ys = allPoints.map(p => p.reward);
  const xMin = Math.min(...xs, 0);
  const xMax = Math.max(...xs, 1);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }
  const yPad = (yMax - yMin) * 0.06;
  yMin -= yPad;
  yMax += yPad;

  const W = 880, H = 460;
  const ml = 60, mr = 24, mt = 24, mb = 50;
  const plotW = W - ml - mr;
  const plotH = H - mt - mb;
  const xScale = x => ml + (x - xMin) / (xMax - xMin || 1) * plotW;
  const yScale = y => mt + (1 - (y - yMin) / (yMax - yMin || 1)) * plotH;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", W);
  svg.setAttribute("height", H);
  svg.classList.add("curve-chart");

  // Y grid + ticks
  const yTickN = 6;
  for (let i = 0; i <= yTickN; i++) {
    const yv = yMin + (yMax - yMin) * i / yTickN;
    const y = yScale(yv);
    const grid = makeAxisLine(ml, y, ml + plotW, y, "grid");
    svg.appendChild(grid);
    svg.appendChild(makeText(ml - 8, y + 3, yv.toFixed(3), "tick-label", "end"));
  }
  // X ticks (integers within range, spaced if too many)
  const xRange = xMax - xMin;
  const xStep = Math.max(1, Math.ceil(xRange / 12));
  for (let i = Math.ceil(xMin); i <= Math.floor(xMax); i += xStep) {
    const x = xScale(i);
    svg.appendChild(makeText(x, mt + plotH + 16, String(i), "tick-label", "middle"));
    const tick = makeAxisLine(x, mt + plotH, x, mt + plotH + 4, "axis");
    svg.appendChild(tick);
  }
  // Axes
  svg.appendChild(makeAxisLine(ml, mt + plotH, ml + plotW, mt + plotH, "axis"));
  svg.appendChild(makeAxisLine(ml, mt, ml, mt + plotH, "axis"));
  // Axis labels
  svg.appendChild(makeText(ml + plotW / 2, mt + plotH + 38, "iteration", "axis-label", "middle"));
  const yLabel = makeText(16, mt + plotH / 2, "reward", "axis-label", "middle");
  yLabel.setAttribute("transform", `rotate(-90, 16, ${mt + plotH / 2})`);
  svg.appendChild(yLabel);

  // Per-island scatter dots, then best-so-far polylines, then markers
  for (const s of series) {
    for (const p of s.points) {
      const c = makeCircle(xScale(p.iter), yScale(p.reward), 3.5, s.color);
      c.setAttribute("stroke", "none");
      c.classList.add("scatter-dot");
      c.dataset.title = `island ${s.id} · iter ${p.iter}\nreward = ${p.reward.toFixed(4)}`;
      svg.appendChild(c);
    }
  }
  for (const s of series) {
    if (s.bestEver.length >= 2) {
      const path = document.createElementNS(SVG_NS, "polyline");
      path.setAttribute(
        "points",
        s.bestEver.map(b => `${xScale(b.iter)},${yScale(b.best)}`).join(" "),
      );
      path.setAttribute("stroke", s.color);
      path.classList.add("best-line");
      svg.appendChild(path);
    }
    for (const b of s.bestEver) {
      const c = makeCircle(xScale(b.iter), yScale(b.best), 4.5, s.color);
      c.classList.add("best-marker");
      c.dataset.title = `island ${s.id} · iter ${b.iter}\nbest so far = ${b.best.toFixed(4)}`;
      svg.appendChild(c);
    }
  }

  // Legend
  let lx = ml + plotW - 130;
  let ly = mt + 12;
  for (const s of series) {
    const dot = makeCircle(lx, ly, 5, s.color);
    dot.setAttribute("stroke", "none");
    svg.appendChild(dot);
    svg.appendChild(makeText(lx + 10, ly + 4, `island ${s.id}`, "legend-text", "start"));
    ly += 18;
  }

  // Tooltip handling
  svg.addEventListener("mousemove", e => {
    if (e.target.dataset && e.target.dataset.title) {
      showTooltip(e, e.target.dataset.title);
    } else {
      hideTooltip();
    }
  });
  svg.addEventListener("mouseleave", hideTooltip);

  root.appendChild(wrapSvgWithExport(svg, "score_curve.svg"));
}

function makeAxisLine(x1, y1, x2, y2, cls) {
  const l = document.createElementNS(SVG_NS, "line");
  l.setAttribute("x1", x1);
  l.setAttribute("y1", y1);
  l.setAttribute("x2", x2);
  l.setAttribute("y2", y2);
  if (cls) l.classList.add(cls);
  return l;
}

function makeText(x, y, text, cls, anchor) {
  const t = document.createElementNS(SVG_NS, "text");
  t.setAttribute("x", x);
  t.setAttribute("y", y);
  if (anchor) t.setAttribute("text-anchor", anchor);
  if (cls) t.classList.add(cls);
  t.textContent = text;
  return t;
}

// ----- token & cost view ----------------------------------------------------

function renderCost(events, islands) {
  const root = document.getElementById("cost-chart");
  root.innerHTML = "";

  // Aggregate tokens & cost per iteration across all islands.
  const iterMap = new Map();
  for (const e of events) {
    if (e.type !== "proposal" && e.type !== "mh_step") continue;
    const it = e.iteration;
    if (!iterMap.has(it)) iterMap.set(it, { iter: it, input: 0, output: 0, cost: 0 });
    const bucket = iterMap.get(it);
    const meta = e.proposal_metadata || {};
    const usage = meta.usage || {};
    bucket.input += (usage.prompt_tokens || 0);
    bucket.output += (usage.completion_tokens || 0);
    bucket.cost += (meta.cost_usd || 0);
  }

  const data = [...iterMap.values()].sort((a, b) => a.iter - b.iter);
  if (data.length === 0) {
    root.innerHTML = "<div style='color:#6b7280; font-size:12px;'>No token data yet.</div>";
    return;
  }

  // Cumulative cost series
  let cumCost = 0;
  for (const d of data) {
    cumCost += d.cost;
    d.cumCost = cumCost;
  }

  // ---- Token bar chart (stacked: input + output) ----
  const tokenDiv = document.createElement("div");
  tokenDiv.className = "cost-section";
  tokenDiv.innerHTML = "<h3 class='cost-title'>Tokens per iteration</h3>";
  root.appendChild(tokenDiv);

  const maxTokens = Math.max(...data.map(d => d.input + d.output), 1);
  const nBars = data.length;
  const W1 = Math.max(600, nBars * 36 + 100);
  const H1 = 320;
  const ml1 = 72, mr1 = 24, mt1 = 20, mb1 = 50;
  const plotW1 = W1 - ml1 - mr1;
  const plotH1 = H1 - mt1 - mb1;
  const barW = Math.min(28, (plotW1 / nBars) * 0.7);
  const barGap = plotW1 / nBars;

  const svg1 = document.createElementNS(SVG_NS, "svg");
  svg1.setAttribute("width", W1);
  svg1.setAttribute("height", H1);
  svg1.classList.add("cost-chart-svg");

  // Y grid
  const yTicks1 = 5;
  for (let i = 0; i <= yTicks1; i++) {
    const val = maxTokens * i / yTicks1;
    const y = mt1 + plotH1 - (val / maxTokens) * plotH1;
    svg1.appendChild(makeAxisLine(ml1, y, ml1 + plotW1, y, "grid"));
    svg1.appendChild(makeText(ml1 - 8, y + 3, fmtTokens(val), "tick-label", "end"));
  }

  // Axes
  svg1.appendChild(makeAxisLine(ml1, mt1 + plotH1, ml1 + plotW1, mt1 + plotH1, "axis"));
  svg1.appendChild(makeAxisLine(ml1, mt1, ml1, mt1 + plotH1, "axis"));

  // Bars
  for (let i = 0; i < nBars; i++) {
    const d = data[i];
    const bx = ml1 + i * barGap + (barGap - barW) / 2;
    const hIn = (d.input / maxTokens) * plotH1;
    const hOut = (d.output / maxTokens) * plotH1;

    // Input tokens (bottom)
    const rIn = makeRect(bx, mt1 + plotH1 - hIn, barW, hIn, "#3b82f6");
    rIn.dataset.title = `iter ${d.iter}\ninput: ${d.input.toLocaleString()} tokens`;
    svg1.appendChild(rIn);

    // Output tokens (stacked on top)
    const rOut = makeRect(bx, mt1 + plotH1 - hIn - hOut, barW, hOut, "#f59e0b");
    rOut.dataset.title = `iter ${d.iter}\noutput: ${d.output.toLocaleString()} tokens`;
    svg1.appendChild(rOut);

    // X label (show every N ticks to avoid crowding)
    const xStep = Math.max(1, Math.ceil(nBars / 20));
    if (i % xStep === 0) {
      svg1.appendChild(makeText(bx + barW / 2, mt1 + plotH1 + 14, String(d.iter), "tick-label", "middle"));
    }
  }

  // Axis labels
  svg1.appendChild(makeText(ml1 + plotW1 / 2, mt1 + plotH1 + 38, "iteration", "axis-label", "middle"));
  const yLab1 = makeText(16, mt1 + plotH1 / 2, "tokens", "axis-label", "middle");
  yLab1.setAttribute("transform", `rotate(-90, 16, ${mt1 + plotH1 / 2})`);
  svg1.appendChild(yLab1);

  // Legend
  const lg1y = mt1 + 6;
  svg1.appendChild(makeRect(ml1 + plotW1 - 160, lg1y, 10, 10, "#3b82f6"));
  svg1.appendChild(makeText(ml1 + plotW1 - 146, lg1y + 9, "input tokens", "legend-text", "start"));
  svg1.appendChild(makeRect(ml1 + plotW1 - 72, lg1y, 10, 10, "#f59e0b"));
  svg1.appendChild(makeText(ml1 + plotW1 - 58, lg1y + 9, "output tokens", "legend-text", "start"));

  // Tooltip
  svg1.addEventListener("mousemove", e => {
    if (e.target.dataset && e.target.dataset.title) showTooltip(e, e.target.dataset.title);
    else hideTooltip();
  });
  svg1.addEventListener("mouseleave", hideTooltip);

  tokenDiv.appendChild(wrapSvgWithExport(svg1, "tokens_per_iteration.svg"));

  // ---- Per-model cumulative cost line chart ----
  // Build per-model cost series
  const modelCostMap = new Map(); // model -> Map(iter -> cost_in_this_iter)
  for (const e of events) {
    if (e.type !== "proposal" && e.type !== "mh_step") continue;
    const meta = e.proposal_metadata || {};
    const model = meta.model || "unknown";
    const costVal = meta.cost_usd || 0;
    if (!modelCostMap.has(model)) modelCostMap.set(model, new Map());
    const mIter = modelCostMap.get(model);
    mIter.set(e.iteration, (mIter.get(e.iteration) || 0) + costVal);
  }

  const allIters = [...new Set(data.map(d => d.iter))].sort((a, b) => a - b);
  const modelNames = [...modelCostMap.keys()].sort();

  const modelSeries = modelNames.map((model, idx) => {
    const iterCost = modelCostMap.get(model);
    let cum = 0;
    const points = allIters.map(it => {
      cum += (iterCost.get(it) || 0);
      return { iter: it, cumCost: cum };
    });
    return {
      model,
      color: MODEL_COLORS[idx % MODEL_COLORS.length],
      points,
      totalCost: cum,
    };
  });

  const costDiv = document.createElement("div");
  costDiv.className = "cost-section";
  costDiv.innerHTML = "<h3 class='cost-title'>Cumulative cost per model (USD)</h3>";
  root.appendChild(costDiv);

  const maxCost = Math.max(...modelSeries.flatMap(s => s.points.map(p => p.cumCost)), 1e-9);
  const iters = data.map(d => d.iter);
  const xMin2 = Math.min(...iters, 0);
  const xMax2 = Math.max(...iters, 1);

  const W2 = 880, H2 = 340;
  const ml2 = 72, mr2 = 24, mt2 = 20, mb2 = 50;
  const plotW2 = W2 - ml2 - mr2;
  const plotH2 = H2 - mt2 - mb2;
  const xScale2 = x => ml2 + (x - xMin2) / (xMax2 - xMin2 || 1) * plotW2;
  const yScale2 = y => mt2 + (1 - y / maxCost) * plotH2;

  const svg2 = document.createElementNS(SVG_NS, "svg");
  svg2.setAttribute("width", W2);
  svg2.setAttribute("height", H2);
  svg2.classList.add("cost-chart-svg");

  // Y grid
  const yTicks2 = 5;
  for (let i = 0; i <= yTicks2; i++) {
    const val = maxCost * i / yTicks2;
    const y = yScale2(val);
    svg2.appendChild(makeAxisLine(ml2, y, ml2 + plotW2, y, "grid"));
    svg2.appendChild(makeText(ml2 - 8, y + 3, "$" + fmtCost(val), "tick-label", "end"));
  }

  // X ticks
  const xRange2 = xMax2 - xMin2;
  const xStep2 = Math.max(1, Math.ceil(xRange2 / 12));
  for (let i = Math.ceil(xMin2); i <= Math.floor(xMax2); i += xStep2) {
    const x = xScale2(i);
    svg2.appendChild(makeText(x, mt2 + plotH2 + 16, String(i), "tick-label", "middle"));
    svg2.appendChild(makeAxisLine(x, mt2 + plotH2, x, mt2 + plotH2 + 4, "axis"));
  }

  // Axes
  svg2.appendChild(makeAxisLine(ml2, mt2 + plotH2, ml2 + plotW2, mt2 + plotH2, "axis"));
  svg2.appendChild(makeAxisLine(ml2, mt2, ml2, mt2 + plotH2, "axis"));

  // Per-model lines and dots
  for (const s of modelSeries) {
    // Area fill
    if (s.points.length >= 2) {
      const areaPath = document.createElementNS(SVG_NS, "polygon");
      const pts = s.points.map(p => `${xScale2(p.iter)},${yScale2(p.cumCost)}`);
      pts.push(`${xScale2(s.points[s.points.length - 1].iter)},${yScale2(0)}`);
      pts.unshift(`${xScale2(s.points[0].iter)},${yScale2(0)}`);
      areaPath.setAttribute("points", pts.join(" "));
      areaPath.setAttribute("fill", s.color);
      areaPath.setAttribute("opacity", 0.08);
      svg2.appendChild(areaPath);
    }

    // Line
    if (s.points.length >= 2) {
      const line = document.createElementNS(SVG_NS, "polyline");
      line.setAttribute("points", s.points.map(p => `${xScale2(p.iter)},${yScale2(p.cumCost)}`).join(" "));
      line.setAttribute("stroke", s.color);
      line.setAttribute("stroke-width", 2);
      line.setAttribute("fill", "none");
      svg2.appendChild(line);
    }

    // Dots
    for (const p of s.points) {
      if (p.cumCost === 0) continue;
      const c = makeCircle(xScale2(p.iter), yScale2(p.cumCost), 3.5, s.color);
      c.setAttribute("stroke", "#fff");
      c.setAttribute("stroke-width", 1);
      c.style.cursor = "pointer";
      c.dataset.title = `${s.model}\niter ${p.iter}\ncumulative: $${fmtCost(p.cumCost)}`;
      svg2.appendChild(c);
    }
  }

  // Axis labels
  svg2.appendChild(makeText(ml2 + plotW2 / 2, mt2 + plotH2 + 38, "iteration", "axis-label", "middle"));
  const yLab2 = makeText(16, mt2 + plotH2 / 2, "cost (USD)", "axis-label", "middle");
  yLab2.setAttribute("transform", `rotate(-90, 16, ${mt2 + plotH2 / 2})`);
  svg2.appendChild(yLab2);

  // Legend (per model) — top-left to avoid overlapping lines
  let lgX = ml2 + 10;
  let lgY = mt2 + 10;
  for (const s of modelSeries) {
    const label = `${s.model}: $${fmtCost(s.totalCost)}`;
    svg2.appendChild(makeRect(lgX, lgY - 5, 10, 10, s.color));
    svg2.appendChild(makeText(lgX + 14, lgY + 4, label, "legend-text", "start"));
    lgY += 16;
  }

  // Total cost annotation
  const totalText = makeText(ml2 + plotW2 - 8, mt2 + 14, `total: $${fmtCost(cumCost)}`, "cost-total", "end");
  svg2.appendChild(totalText);

  // Tooltip
  svg2.addEventListener("mousemove", e => {
    if (e.target.dataset && e.target.dataset.title) showTooltip(e, e.target.dataset.title);
    else hideTooltip();
  });
  svg2.addEventListener("mouseleave", hideTooltip);

  costDiv.appendChild(wrapSvgWithExport(svg2, "cumulative_cost_per_model.svg"));

  // ---- Best score per model ----
  const modelScoreMap = new Map(); // model -> { best, calls, improved }
  for (const e of events) {
    if (e.type !== "proposal" && e.type !== "mh_step") continue;
    const meta = e.proposal_metadata || {};
    const model = meta.model || "unknown";
    if (!modelScoreMap.has(model)) modelScoreMap.set(model, { best: -Infinity, calls: 0, improved: 0 });
    const entry = modelScoreMap.get(model);
    entry.calls++;
    // Only count actually evaluated proposals (not skipped/unchanged)
    if (!e.skipped && typeof e.child_reward === "number" && isFinite(e.child_reward)) {
      if (e.child_reward > entry.best) entry.best = e.child_reward;
    }
    if (e.accepted) entry.improved++;
  }

  const bestScoreDiv = document.createElement("div");
  bestScoreDiv.className = "cost-section";
  bestScoreDiv.innerHTML = "<h3 class='cost-title'>Best score per model</h3>";
  const scoreTable = document.createElement("table");
  scoreTable.className = "model-score-table";
  scoreTable.innerHTML = `<tr><th>Model</th><th>Best Score</th><th>Calls</th><th>Improved</th><th>Improve Rate</th></tr>`;
  for (const model of [...modelScoreMap.keys()].sort()) {
    const s = modelScoreMap.get(model);
    const bestStr = s.best > -Infinity ? s.best.toFixed(4) : "N/A";
    const rateStr = s.calls > 0 ? (s.improved / s.calls * 100).toFixed(1) + "%" : "N/A";
    scoreTable.innerHTML += `<tr><td>${model}</td><td>${bestStr}</td><td>${s.calls}</td><td>${s.improved}</td><td>${rateStr}</td></tr>`;
  }
  bestScoreDiv.appendChild(scoreTable);
  root.appendChild(bestScoreDiv);

  // ---- Summary table ----
  const totalInput = data.reduce((s, d) => s + d.input, 0);
  const totalOutput = data.reduce((s, d) => s + d.output, 0);
  const summaryDiv = document.createElement("div");
  summaryDiv.className = "cost-summary";
  summaryDiv.innerHTML = `
    <table>
      <tr><th>Total input tokens</th><td>${totalInput.toLocaleString()}</td></tr>
      <tr><th>Total output tokens</th><td>${totalOutput.toLocaleString()}</td></tr>
      <tr><th>Total tokens</th><td>${(totalInput + totalOutput).toLocaleString()}</td></tr>
      <tr><th>Total cost</th><td>$${fmtCost(cumCost)}</td></tr>
    </table>
  `;
  root.appendChild(summaryDiv);
}

function makeRect(x, y, w, h, fill) {
  const r = document.createElementNS(SVG_NS, "rect");
  r.setAttribute("x", x);
  r.setAttribute("y", y);
  r.setAttribute("width", w);
  r.setAttribute("height", Math.max(0, h));
  r.setAttribute("fill", fill);
  r.setAttribute("rx", 2);
  return r;
}

function fmtTokens(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}

function fmtCost(c) {
  if (c >= 1) return c.toFixed(2);
  if (c >= 0.01) return c.toFixed(3);
  return c.toFixed(4);
}

// ----- SVG export -----------------------------------------------------------

// ----- kernel tab -----------------------------------------------------------

const KERNEL_COLORS = {
  diff_no_inspo: "#3b82f6",
  diff_with_inspo: "#60a5fa",
  rewrite_no_inspo: "#f59e0b",
  rewrite_with_inspo: "#fbbf24",
};

function renderKernel(events, islands) {
  const root = document.getElementById("kernel-chart");
  root.innerHTML = "";

  // Collect per-iteration per-kernel counts & improvements
  const iterMap = new Map(); // iter -> { kernel -> { count, improved } }
  for (const e of events) {
    if (e.type !== "proposal" && e.type !== "mh_step") continue;
    const meta = e.proposal_metadata || {};
    const kernel = meta.kernel || "unknown";
    const it = e.iteration;
    if (!iterMap.has(it)) iterMap.set(it, {});
    const bucket = iterMap.get(it);
    if (!bucket[kernel]) bucket[kernel] = { count: 0, improved: 0 };
    bucket[kernel].count++;
    // Use "improved" field if present, else fall back to "accepted"
    if (e.improved || (e.improved === undefined && e.accepted)) {
      bucket[kernel].improved++;
    }
  }

  const data = [...iterMap.entries()]
    .sort(([a], [b]) => a - b)
    .map(([iter, buckets]) => ({ iter, buckets }));

  if (data.length === 0) {
    root.innerHTML = "<div style='color:#6b7280; font-size:12px;'>No kernel data yet.</div>";
    return;
  }

  const allKernels = [...new Set(data.flatMap(d => Object.keys(d.buckets)))].sort();

  // ---- 1. Kernel selection count (stacked bar) ----
  const selDiv = document.createElement("div");
  selDiv.className = "cost-section";
  selDiv.innerHTML = "<h3 class='cost-title'>Kernel selection per iteration</h3>";
  root.appendChild(selDiv);

  const maxCount = Math.max(...data.map(d =>
    Object.values(d.buckets).reduce((s, b) => s + b.count, 0)
  ), 1);
  const nBars = data.length;
  const W1 = Math.max(600, nBars * 36 + 100);
  const H1 = 300;
  const ml1 = 72, mr1 = 24, mt1 = 20, mb1 = 50;
  const plotW1 = W1 - ml1 - mr1;
  const plotH1 = H1 - mt1 - mb1;
  const barW1 = Math.min(28, (plotW1 / nBars) * 0.7);
  const barGap1 = plotW1 / nBars;

  const svg1 = document.createElementNS(SVG_NS, "svg");
  svg1.setAttribute("width", W1);
  svg1.setAttribute("height", H1);
  svg1.classList.add("cost-chart-svg");

  // Y grid
  for (let i = 0; i <= 5; i++) {
    const val = maxCount * i / 5;
    const y = mt1 + plotH1 - (val / maxCount) * plotH1;
    svg1.appendChild(makeAxisLine(ml1, y, ml1 + plotW1, y, "grid"));
    svg1.appendChild(makeText(ml1 - 8, y + 3, String(Math.round(val)), "tick-label", "end"));
  }
  svg1.appendChild(makeAxisLine(ml1, mt1 + plotH1, ml1 + plotW1, mt1 + plotH1, "axis"));
  svg1.appendChild(makeAxisLine(ml1, mt1, ml1, mt1 + plotH1, "axis"));

  for (let i = 0; i < nBars; i++) {
    const d = data[i];
    const bx = ml1 + i * barGap1 + (barGap1 - barW1) / 2;
    let cumH = 0;
    for (const kernel of allKernels) {
      const cnt = (d.buckets[kernel] || {}).count || 0;
      if (cnt === 0) continue;
      const h = (cnt / maxCount) * plotH1;
      const color = KERNEL_COLORS[kernel] || "#9ca3af";
      const r = makeRect(bx, mt1 + plotH1 - cumH - h, barW1, h, color);
      r.dataset.title = `iter ${d.iter}\n${kernel}: ${cnt}`;
      svg1.appendChild(r);
      cumH += h;
    }
    const xStep = Math.max(1, Math.ceil(nBars / 20));
    if (i % xStep === 0) {
      svg1.appendChild(makeText(bx + barW1 / 2, mt1 + plotH1 + 14, String(d.iter), "tick-label", "middle"));
    }
  }

  svg1.appendChild(makeText(ml1 + plotW1 / 2, mt1 + plotH1 + 38, "iteration", "axis-label", "middle"));
  const yLab1 = makeText(16, mt1 + plotH1 / 2, "count", "axis-label", "middle");
  yLab1.setAttribute("transform", `rotate(-90, 16, ${mt1 + plotH1 / 2})`);
  svg1.appendChild(yLab1);

  // Legend
  let lgX = ml1 + plotW1 - 200, lgY = mt1 + 6;
  for (const kernel of allKernels) {
    const color = KERNEL_COLORS[kernel] || "#9ca3af";
    svg1.appendChild(makeRect(lgX, lgY - 5, 10, 10, color));
    svg1.appendChild(makeText(lgX + 14, lgY + 4, kernel, "legend-text", "start"));
    lgY += 16;
  }

  svg1.addEventListener("mousemove", e => {
    if (e.target.dataset && e.target.dataset.title) showTooltip(e, e.target.dataset.title);
    else hideTooltip();
  });
  svg1.addEventListener("mouseleave", hideTooltip);
  selDiv.appendChild(wrapSvgWithExport(svg1, "kernel_selection.svg"));

  // ---- 2. Per-kernel improvement rate (line chart) ----
  const rateDiv = document.createElement("div");
  rateDiv.className = "cost-section";
  rateDiv.innerHTML = "<h3 class='cost-title'>Kernel improvement rate per iteration</h3>";
  root.appendChild(rateDiv);

  const iters = data.map(d => d.iter);
  const xMin = Math.min(...iters, 0);
  const xMax = Math.max(...iters, 1);
  const W2 = 880, H2 = 340;
  const ml2 = 72, mr2 = 24, mt2 = 20, mb2 = 50;
  const plotW2 = W2 - ml2 - mr2;
  const plotH2 = H2 - mt2 - mb2;
  const xScale = x => ml2 + (x - xMin) / (xMax - xMin || 1) * plotW2;
  const yScale = y => mt2 + (1 - y) * plotH2; // y in [0, 1]

  const svg2 = document.createElementNS(SVG_NS, "svg");
  svg2.setAttribute("width", W2);
  svg2.setAttribute("height", H2);
  svg2.classList.add("cost-chart-svg");

  // Y grid (0% to 100%)
  for (let i = 0; i <= 5; i++) {
    const val = i / 5;
    const y = yScale(val);
    svg2.appendChild(makeAxisLine(ml2, y, ml2 + plotW2, y, "grid"));
    svg2.appendChild(makeText(ml2 - 8, y + 3, (val * 100).toFixed(0) + "%", "tick-label", "end"));
  }
  // X ticks
  const xStep2 = Math.max(1, Math.ceil((xMax - xMin) / 12));
  for (let i = Math.ceil(xMin); i <= Math.floor(xMax); i += xStep2) {
    svg2.appendChild(makeText(xScale(i), mt2 + plotH2 + 16, String(i), "tick-label", "middle"));
    svg2.appendChild(makeAxisLine(xScale(i), mt2 + plotH2, xScale(i), mt2 + plotH2 + 4, "axis"));
  }
  svg2.appendChild(makeAxisLine(ml2, mt2 + plotH2, ml2 + plotW2, mt2 + plotH2, "axis"));
  svg2.appendChild(makeAxisLine(ml2, mt2, ml2, mt2 + plotH2, "axis"));

  for (const kernel of allKernels) {
    const color = KERNEL_COLORS[kernel] || "#9ca3af";
    const pts = [];
    for (const d of data) {
      const b = d.buckets[kernel];
      if (!b || b.count === 0) continue;
      const rate = b.improved / b.count;
      pts.push({ iter: d.iter, rate });
    }
    if (pts.length >= 2) {
      const line = document.createElementNS(SVG_NS, "polyline");
      line.setAttribute("points", pts.map(p => `${xScale(p.iter)},${yScale(p.rate)}`).join(" "));
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", 2);
      line.setAttribute("fill", "none");
      svg2.appendChild(line);
    }
    for (const p of pts) {
      const c = makeCircle(xScale(p.iter), yScale(p.rate), 3.5, color);
      c.setAttribute("stroke", "#fff");
      c.setAttribute("stroke-width", 1);
      c.dataset.title = `${kernel}\niter ${p.iter}\nrate = ${(p.rate * 100).toFixed(1)}%`;
      svg2.appendChild(c);
    }
  }

  svg2.appendChild(makeText(ml2 + plotW2 / 2, mt2 + plotH2 + 38, "iteration", "axis-label", "middle"));
  const yLab2 = makeText(16, mt2 + plotH2 / 2, "improve rate", "axis-label", "middle");
  yLab2.setAttribute("transform", `rotate(-90, 16, ${mt2 + plotH2 / 2})`);
  svg2.appendChild(yLab2);

  // Legend
  let lgX2 = ml2 + 10, lgY2 = mt2 + 10;
  for (const kernel of allKernels) {
    const color = KERNEL_COLORS[kernel] || "#9ca3af";
    svg2.appendChild(makeRect(lgX2, lgY2 - 5, 10, 10, color));
    svg2.appendChild(makeText(lgX2 + 14, lgY2 + 4, kernel, "legend-text", "start"));
    lgY2 += 16;
  }

  svg2.addEventListener("mousemove", e => {
    if (e.target.dataset && e.target.dataset.title) showTooltip(e, e.target.dataset.title);
    else hideTooltip();
  });
  svg2.addEventListener("mouseleave", hideTooltip);
  rateDiv.appendChild(wrapSvgWithExport(svg2, "kernel_improve_rate.svg"));

  // ---- 3. Thompson Sampling posterior (from step_summary.kernel_stats) ----
  const tsSeries = []; // { iter, stats: { kernel: {alpha, beta, mean} } }
  for (const e of events) {
    if (e.type !== "step_summary" || !e.kernel_stats) continue;
    tsSeries.push({ iter: e.iteration, stats: e.kernel_stats });
  }

  if (tsSeries.length > 0) {
    const tsDiv = document.createElement("div");
    tsDiv.className = "cost-section";
    tsDiv.innerHTML = "<h3 class='cost-title'>Adaptive kernel posterior (Thompson Sampling mean)</h3>";
    root.appendChild(tsDiv);

    const tsKernels = [...new Set(tsSeries.flatMap(t => Object.keys(t.stats)))].sort();
    const W3 = 880, H3 = 340;
    const ml3 = 72, mr3 = 24, mt3 = 20, mb3 = 50;
    const plotW3 = W3 - ml3 - mr3;
    const plotH3 = H3 - mt3 - mb3;
    const tsIters = tsSeries.map(t => t.iter);
    const txMin = Math.min(...tsIters, 0);
    const txMax = Math.max(...tsIters, 1);
    const txScale = x => ml3 + (x - txMin) / (txMax - txMin || 1) * plotW3;
    const tyScale = y => mt3 + (1 - y) * plotH3;

    const svg3 = document.createElementNS(SVG_NS, "svg");
    svg3.setAttribute("width", W3);
    svg3.setAttribute("height", H3);
    svg3.classList.add("cost-chart-svg");

    for (let i = 0; i <= 5; i++) {
      const val = i / 5;
      const y = tyScale(val);
      svg3.appendChild(makeAxisLine(ml3, y, ml3 + plotW3, y, "grid"));
      svg3.appendChild(makeText(ml3 - 8, y + 3, val.toFixed(1), "tick-label", "end"));
    }
    const txStep = Math.max(1, Math.ceil((txMax - txMin) / 12));
    for (let i = Math.ceil(txMin); i <= Math.floor(txMax); i += txStep) {
      svg3.appendChild(makeText(txScale(i), mt3 + plotH3 + 16, String(i), "tick-label", "middle"));
      svg3.appendChild(makeAxisLine(txScale(i), mt3 + plotH3, txScale(i), mt3 + plotH3 + 4, "axis"));
    }
    svg3.appendChild(makeAxisLine(ml3, mt3 + plotH3, ml3 + plotW3, mt3 + plotH3, "axis"));
    svg3.appendChild(makeAxisLine(ml3, mt3, ml3, mt3 + plotH3, "axis"));

    for (const kernel of tsKernels) {
      const color = KERNEL_COLORS[kernel] || "#9ca3af";
      const pts = tsSeries
        .filter(t => t.stats[kernel])
        .map(t => ({ iter: t.iter, mean: t.stats[kernel].mean }));
      if (pts.length >= 2) {
        const line = document.createElementNS(SVG_NS, "polyline");
        line.setAttribute("points", pts.map(p => `${txScale(p.iter)},${tyScale(p.mean)}`).join(" "));
        line.setAttribute("stroke", color);
        line.setAttribute("stroke-width", 2);
        line.setAttribute("fill", "none");
        svg3.appendChild(line);
      }
      for (const p of pts) {
        const c = makeCircle(txScale(p.iter), tyScale(p.mean), 3.5, color);
        c.setAttribute("stroke", "#fff");
        c.setAttribute("stroke-width", 1);
        c.dataset.title = `${kernel}\niter ${p.iter}\nposterior mean = ${p.mean.toFixed(3)}`;
        svg3.appendChild(c);
      }
    }

    svg3.appendChild(makeText(ml3 + plotW3 / 2, mt3 + plotH3 + 38, "iteration", "axis-label", "middle"));
    const yLab3 = makeText(16, mt3 + plotH3 / 2, "posterior mean", "axis-label", "middle");
    yLab3.setAttribute("transform", `rotate(-90, 16, ${mt3 + plotH3 / 2})`);
    svg3.appendChild(yLab3);

    let lgX3 = ml3 + 10, lgY3 = mt3 + 10;
    for (const kernel of tsKernels) {
      const color = KERNEL_COLORS[kernel] || "#9ca3af";
      svg3.appendChild(makeRect(lgX3, lgY3 - 5, 10, 10, color));
      svg3.appendChild(makeText(lgX3 + 14, lgY3 + 4, kernel, "legend-text", "start"));
      lgY3 += 16;
    }

    svg3.addEventListener("mousemove", e => {
      if (e.target.dataset && e.target.dataset.title) showTooltip(e, e.target.dataset.title);
      else hideTooltip();
    });
    svg3.addEventListener("mouseleave", hideTooltip);
    tsDiv.appendChild(wrapSvgWithExport(svg3, "kernel_posterior.svg"));
  }

  // ---- 4. Summary table ----
  const summaryDiv = document.createElement("div");
  summaryDiv.className = "cost-section";
  summaryDiv.innerHTML = "<h3 class='cost-title'>Kernel summary</h3>";
  const table = document.createElement("table");
  table.className = "model-score-table";
  table.innerHTML = "<tr><th>Kernel</th><th>Selected</th><th>Improved</th><th>Improve Rate</th></tr>";
  const totals = {};
  for (const d of data) {
    for (const [k, b] of Object.entries(d.buckets)) {
      if (!totals[k]) totals[k] = { count: 0, improved: 0 };
      totals[k].count += b.count;
      totals[k].improved += b.improved;
    }
  }
  for (const kernel of allKernels) {
    const t = totals[kernel] || { count: 0, improved: 0 };
    const rate = t.count > 0 ? (t.improved / t.count * 100).toFixed(1) + "%" : "N/A";
    table.innerHTML += `<tr><td>${kernel}</td><td>${t.count}</td><td>${t.improved}</td><td>${rate}</td></tr>`;
  }
  summaryDiv.appendChild(table);
  root.appendChild(summaryDiv);
}

// ----- SVG export -----------------------------------------------------------

function downloadSVG(svgEl, filename) {
  const clone = svgEl.cloneNode(true);
  clone.setAttribute("xmlns", SVG_NS);
  clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");

  // Embed page CSS so the exported SVG renders correctly standalone
  const css = getAllCSS();
  const styleEl = document.createElementNS(SVG_NS, "style");
  styleEl.textContent = css;
  clone.insertBefore(styleEl, clone.firstChild);

  // White background
  const bg = document.createElementNS(SVG_NS, "rect");
  bg.setAttribute("width", clone.getAttribute("width") || "100%");
  bg.setAttribute("height", clone.getAttribute("height") || "100%");
  bg.setAttribute("fill", "#fff");
  clone.insertBefore(bg, clone.firstChild);

  const serializer = new XMLSerializer();
  const svgStr =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    serializer.serializeToString(clone);
  const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function getAllCSS() {
  const parts = [];
  for (const sheet of document.styleSheets) {
    try {
      for (const rule of sheet.cssRules) parts.push(rule.cssText);
    } catch (e) { /* cross-origin sheet */ }
  }
  return parts.join("\n");
}

function wrapSvgWithExport(svg, filename) {
  const wrap = document.createElement("div");
  wrap.className = "svg-export-wrap";
  wrap.appendChild(svg);
  const btn = document.createElement("button");
  btn.className = "export-svg-btn";
  btn.textContent = "\u2193 SVG";
  btn.title = "Export as SVG file";
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    downloadSVG(svg, filename);
  });
  wrap.appendChild(btn);
  return wrap;
}

function downloadAllFlowSVG() {
  if (!CURRENT) return;

  const pad = 20;
  const statsW = 190;
  const titleH = 32;
  const iterGap = 14;
  const islandGap = 28;
  const font = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif";

  // Collect sections from the DOM
  const sections = [];
  let y = pad;
  let maxSvgW = 0;

  const islandDivs = document.querySelectorAll("#islands .island");
  for (const islDiv of islandDivs) {
    const titleText = islDiv.querySelector("h2").textContent.trim();
    sections.push({ type: "title", text: titleText, y: y + 18 });
    y += titleH;

    const iterDivs = islDiv.querySelectorAll(".iter");
    for (const iterDiv of iterDivs) {
      const svg = iterDiv.querySelector("svg.flow");
      if (!svg) continue;
      const w = parseFloat(svg.getAttribute("width")) || 400;
      const h = parseFloat(svg.getAttribute("height")) || 200;
      maxSvgW = Math.max(maxSvgW, w);

      const statsDiv = iterDiv.querySelector(".stats");
      const statsLines = [];
      if (statsDiv) {
        for (const child of statsDiv.children) {
          statsLines.push(child.textContent.trim());
        }
      }

      sections.push({ type: "iter", svg, w, h, statsLines, y });
      y += h + iterGap;
    }
    y += islandGap;
  }

  if (sections.filter(s => s.type === "iter").length === 0) return;

  const totalW = pad + statsW + maxSvgW + pad;
  const totalH = y + pad;

  // Build combined SVG
  const combined = document.createElementNS(SVG_NS, "svg");
  combined.setAttribute("xmlns", SVG_NS);
  combined.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  combined.setAttribute("width", totalW);
  combined.setAttribute("height", totalH);

  // White background
  const bg = document.createElementNS(SVG_NS, "rect");
  bg.setAttribute("width", totalW);
  bg.setAttribute("height", totalH);
  bg.setAttribute("fill", "#fff");
  combined.appendChild(bg);

  // Embed CSS
  const styleEl = document.createElementNS(SVG_NS, "style");
  styleEl.textContent = getAllCSS();
  combined.appendChild(styleEl);

  // Shared defs for arrow markers
  const combinedDefs = document.createElementNS(SVG_NS, "defs");
  combined.appendChild(combinedDefs);

  let prevWasIter = false;
  for (const sec of sections) {
    if (sec.type === "title") {
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("x", pad);
      t.setAttribute("y", sec.y);
      t.setAttribute("font-size", "15");
      t.setAttribute("font-weight", "600");
      t.setAttribute("fill", "#1c1f24");
      t.setAttribute("font-family", font);
      t.textContent = sec.text;
      combined.appendChild(t);
      prevWasIter = false;
    } else {
      // Dashed separator between iterations
      if (prevWasIter) {
        const sep = document.createElementNS(SVG_NS, "line");
        sep.setAttribute("x1", pad);
        sep.setAttribute("x2", pad + statsW + sec.w);
        sep.setAttribute("y1", sec.y - iterGap / 2);
        sep.setAttribute("y2", sec.y - iterGap / 2);
        sep.setAttribute("stroke", "#e5e7eb");
        sep.setAttribute("stroke-dasharray", "4,3");
        combined.appendChild(sep);
      }

      // Stats text column
      const statsG = document.createElementNS(SVG_NS, "g");
      statsG.setAttribute("transform", `translate(${pad}, ${sec.y})`);
      let sy = 14;
      for (let i = 0; i < sec.statsLines.length; i++) {
        const t = document.createElementNS(SVG_NS, "text");
        t.setAttribute("x", 0);
        t.setAttribute("y", sy);
        t.setAttribute("font-family", font);
        if (i === 0) {
          t.setAttribute("font-size", "14");
          t.setAttribute("font-weight", "700");
          t.setAttribute("fill", "#1c1f24");
        } else {
          t.setAttribute("font-size", "11");
          t.setAttribute("fill", "#4b5563");
        }
        t.textContent = sec.statsLines[i];
        statsG.appendChild(t);
        sy += i === 0 ? 18 : 14;
      }
      combined.appendChild(statsG);

      // Flow SVG content — hoist <defs> to combined level
      const svgG = document.createElementNS(SVG_NS, "g");
      svgG.setAttribute("transform", `translate(${pad + statsW}, ${sec.y})`);
      const clone = sec.svg.cloneNode(true);
      const defs = clone.querySelector("defs");
      if (defs) {
        while (defs.firstChild) combinedDefs.appendChild(defs.firstChild);
        defs.remove();
      }
      while (clone.firstChild) svgG.appendChild(clone.firstChild);
      combined.appendChild(svgG);
      prevWasIter = true;
    }
  }

  // Serialize & download
  const serializer = new XMLSerializer();
  const svgStr =
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    serializer.serializeToString(combined);
  const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "flow_all.svg";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
