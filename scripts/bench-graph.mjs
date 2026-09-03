#!/usr/bin/env node
/**
 * One-shot benchmark for shared/graph.js (plan F2 acceptance: ~1e6 gaussians
 * in < 3 s on a laptop). Generates a uniform random cloud; no files are read.
 *
 *   node scripts/bench-graph.mjs [N=1000000] [repeats=2] [--out artifacts/bench/graph.json]
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { buildSuperpointGraph } from "../shared/graph.js";
import { mulberry32 } from "../shared/synthetic.js";

const args = process.argv.slice(2);
const positional = args.filter((a) => !a.startsWith("--"));
const N = Number(positional[0] || 1_000_000);
const repeats = Number(positional[1] || 2);
const outIdx = args.indexOf("--out");
const outPath = outIdx >= 0 ? args[outIdx + 1] : "";
const budgetMs = Number(process.env.GRAPH_BUDGET_MS || 3000);

if (!Number.isInteger(N) || N <= 0) throw new Error(`N inválido: ${positional[0]}`);

const rnd = mulberry32(7);
const gaussians = new Float32Array(N * 12);
const colors = new Float32Array(N * 3);
const spacing = Math.cbrt(1 / N);
for (let i = 0; i < N; i++) {
  const o = i * 12;
  gaussians[o] = rnd();
  gaussians[o + 1] = rnd();
  gaussians[o + 2] = rnd();
  gaussians[o + 3] = 0.8;
  gaussians[o + 4] = spacing * (0.5 + rnd());
  gaussians[o + 5] = spacing * (0.5 + rnd());
  gaussians[o + 6] = spacing * (0.5 + rnd());
  gaussians[o + 8] = 1;
  colors[i * 3] = rnd();
  colors[i * 3 + 1] = rnd();
  colors[i * 3 + 2] = rnd();
}

const runs = [];
for (let r = 0; r < repeats; r++) {
  const t0 = performance.now();
  const g = buildSuperpointGraph(gaussians, colors);
  const ms = performance.now() - t0;
  const stats = Object.fromEntries(Object.entries(g.stats).map(([k, v]) => [k, typeof v === "number" ? Math.round(v * 10) / 10 : v]));
  runs.push({ run: r + 1, ms: Math.round(ms), superpoints: g.superpointCount, stats });
  console.log(
    `[bench-graph] ${N.toLocaleString("es-ES")} gaussianas · ejecución ${r + 1}/${repeats}: ${ms.toFixed(0)} ms ` +
      `(rejilla ${stats.msGrid}, kNN ${stats.msKnn}, simetría ${stats.msSymmetrize}, pesos ${stats.msWeights}, ` +
      `componentes ${stats.msComponents}) · ${g.superpointCount} superpuntos · grado medio ${stats.avgDegree}`
  );
}
const best = Math.min(...runs.map((r) => r.ms));
const ok = best < budgetMs;
console.log(`[bench-graph] mejor tiempo ${best} ms · presupuesto ${budgetMs} ms → ${ok ? "OK" : "SUPERA EL PRESUPUESTO"}`);
if (outPath) {
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, JSON.stringify({ n: N, budgetMs, bestMs: best, ok, node: process.version, runs }, null, 2) + "\n");
  console.log(`[bench-graph] resultados en ${outPath}`);
}
process.exitCode = ok ? 0 : 1;
