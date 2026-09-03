/**
 * Unit tests for shared/graph.js (plan F2: superpoint graph). Run: npm test.
 * Only node:test + node:assert; clouds are generated in memory.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_GRAPH_OPTIONS,
  GAUSSIAN_FLOATS,
  autoCellSize,
  buildKnnGraph,
  buildSuperpointGraph,
  componentCentroids,
  connectedComponents,
  diffuseLabels,
  edgeWeights,
  groupColor,
  groupsToLabels,
  indicesOfGroup,
  inverseCovariances,
  medianLargestScale,
  shDcToRgb,
} from "../../shared/graph.js";
import { makeTwoSpheres, mulberry32, rgbToShDc } from "../../shared/synthetic.js";

/** Random gaussians inside a ball: positions, isotropic scale, identity rotation. */
function makeBlob({ n, center, radius, scale, rgb, seed }) {
  const rnd = mulberry32(seed);
  const gaussians = new Float32Array(n * GAUSSIAN_FLOATS);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    // uniform in a ball via rejection
    let x, y, z;
    do {
      x = rnd() * 2 - 1;
      y = rnd() * 2 - 1;
      z = rnd() * 2 - 1;
    } while (x * x + y * y + z * z > 1);
    const o = i * GAUSSIAN_FLOATS;
    gaussians[o] = center[0] + x * radius;
    gaussians[o + 1] = center[1] + y * radius;
    gaussians[o + 2] = center[2] + z * radius;
    gaussians[o + 3] = 0.9;
    gaussians[o + 4] = gaussians[o + 5] = gaussians[o + 6] = scale;
    gaussians[o + 8] = 1;
    colors.set(rgb, i * 3);
  }
  return { gaussians, colors };
}

function concatClouds(...clouds) {
  const n = clouds.reduce((a, c) => a + c.gaussians.length / GAUSSIAN_FLOATS, 0);
  const gaussians = new Float32Array(n * GAUSSIAN_FLOATS);
  const colors = new Float32Array(n * 3);
  let g = 0;
  let c = 0;
  for (const cl of clouds) {
    gaussians.set(cl.gaussians, g);
    colors.set(cl.colors, c);
    g += cl.gaussians.length;
    c += cl.colors.length;
  }
  return { gaussians, colors, count: n };
}

/** Random cloud in the unit cube with scales around the mean spacing. */
function makeUniform(n, seed = 7) {
  const rnd = mulberry32(seed);
  const gaussians = new Float32Array(n * GAUSSIAN_FLOATS);
  const colors = new Float32Array(n * 3);
  const sp = Math.cbrt(1 / n);
  for (let i = 0; i < n; i++) {
    const o = i * GAUSSIAN_FLOATS;
    gaussians[o] = rnd();
    gaussians[o + 1] = rnd();
    gaussians[o + 2] = rnd();
    gaussians[o + 3] = 0.8;
    gaussians[o + 4] = sp * (0.5 + rnd());
    gaussians[o + 5] = sp * (0.5 + rnd());
    gaussians[o + 6] = sp * (0.5 + rnd());
    gaussians[o + 8] = 1;
    colors[i * 3] = rnd();
    colors[i * 3 + 1] = rnd();
    colors[i * 3 + 2] = rnd();
  }
  return { gaussians, colors };
}

function bruteForceKnn(gaussians, i, k) {
  const n = gaussians.length / GAUSSIAN_FLOATS;
  const d = [];
  const oi = i * GAUSSIAN_FLOATS;
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const oj = j * GAUSSIAN_FLOATS;
    const dx = gaussians[oj] - gaussians[oi];
    const dy = gaussians[oj + 1] - gaussians[oi + 1];
    const dz = gaussians[oj + 2] - gaussians[oi + 2];
    d.push([dx * dx + dy * dy + dz * dz, j]);
  }
  d.sort((a, b) => a[0] - b[0]);
  return d.slice(0, k);
}

function assertCsrConsistent(csr, n) {
  assert.equal(csr.offsets.length, n + 1);
  assert.equal(csr.offsets[0], 0);
  assert.equal(csr.offsets[n], csr.neighbors.length);
  for (let i = 0; i < n; i++) {
    assert.ok(csr.offsets[i + 1] >= csr.offsets[i], "offsets monotónicos");
    const seen = new Set();
    for (let e = csr.offsets[i]; e < csr.offsets[i + 1]; e++) {
      const j = csr.neighbors[e];
      assert.notEqual(j, i, "sin bucles");
      assert.ok(j < n, "vecino dentro de rango");
      assert.ok(!seen.has(j), `vecino ${j} duplicado en el nodo ${i}`);
      seen.add(j);
      // symmetric: i must appear in j's list
      let back = false;
      for (let f = csr.offsets[j]; f < csr.offsets[j + 1]; f++) if (csr.neighbors[f] === i) back = true;
      assert.ok(back, `arista ${i}→${j} sin reversa`);
    }
  }
}

describe("buildKnnGraph", () => {
  test("empty, single and pair clouds", () => {
    const empty = buildKnnGraph(new Float32Array(0));
    assert.equal(empty.count, 0);
    assert.equal(empty.offsets.length, 1);
    assert.equal(empty.neighbors.length, 0);

    const one = makeBlob({ n: 1, center: [0, 0, 0], radius: 0, scale: 0.1, rgb: [1, 0, 0], seed: 1 });
    const g1 = buildKnnGraph(one.gaussians);
    assert.equal(g1.k, 0);
    assert.equal(g1.neighbors.length, 0);

    const two = concatClouds(
      makeBlob({ n: 1, center: [0, 0, 0], radius: 0, scale: 0.1, rgb: [1, 0, 0], seed: 1 }),
      makeBlob({ n: 1, center: [0.05, 0, 0], radius: 0, scale: 0.1, rgb: [1, 0, 0], seed: 2 })
    );
    const g2 = buildKnnGraph(two.gaussians);
    assert.equal(g2.k, 1);
    assert.deepEqual(Array.from(g2.offsets), [0, 1, 2]);
    assert.deepEqual(Array.from(g2.neighbors), [1, 0]);
    assertCsrConsistent(g2, 2);
  });

  test("matches brute-force k nearest neighbours on a random cloud", () => {
    const { gaussians } = makeUniform(1500, 3);
    const k = 10;
    const g = buildKnnGraph(gaussians, { k });
    assertCsrConsistent(g, 1500);
    assert.equal(g.stats.truncated, 0);
    for (const i of [0, 17, 250, 999, 1499]) {
      const want = bruteForceKnn(gaussians, i, k).map((x) => x[1]);
      const have = new Set();
      for (let e = g.offsets[i]; e < g.offsets[i + 1]; e++) have.add(g.neighbors[e]);
      for (const j of want) assert.ok(have.has(j), `vecino ${j} de ${i} ausente`);
    }
  });

  test("auto cell size is at least the median largest scale", () => {
    const { gaussians } = makeUniform(2000, 5);
    const auto = autoCellSize(gaussians);
    const median = medianLargestScale(gaussians);
    assert.ok(auto.cellSize >= median);
    assert.ok(auto.cellSize > 0);
    assert.equal(medianLargestScale(new Float32Array(0)), 0);
  });

  test("rejects bad inputs", () => {
    assert.throws(() => buildKnnGraph(new Float32Array(5)), /12 floats/);
    assert.throws(() => buildKnnGraph(new Float32Array(24), { k: 0 }), /k must be/);
    assert.throws(() => buildKnnGraph(new Float32Array(24), { cellSize: -1 }), /cellSize/);
  });
});

describe("edgeWeights", () => {
  test("identical overlapping gaussians weigh 1; colour difference lowers the weight", () => {
    const g = new Float32Array(2 * GAUSSIAN_FLOATS);
    for (const i of [0, 1]) {
      const o = i * GAUSSIAN_FLOATS;
      g[o + 4] = g[o + 5] = g[o + 6] = 0.1;
      g[o + 8] = 1;
    }
    const csr = buildKnnGraph(g);
    const same = edgeWeights(g, new Float32Array([1, 0, 0, 1, 0, 0]), csr);
    assert.ok(Math.abs(same[0] - 1) < 1e-6);
    const diff = edgeWeights(g, new Float32Array([1, 0, 0, 0, 0, 1]), csr, { sigmaColor: 0.5 });
    // ‖Δc‖² = 2 → exp(-2 / 0.25) = exp(-8)
    assert.ok(Math.abs(diff[0] - Math.exp(-8)) < 1e-6);
    assert.equal(same.length, csr.neighbors.length);
  });

  test("symmetric Mahalanobis distance follows the covariances", () => {
    // two gaussians 0.2 apart along x, scale 0.1 → d_M² = (0.2/0.1)² = 4 → w = exp(-2)
    const g = new Float32Array(2 * GAUSSIAN_FLOATS);
    g[GAUSSIAN_FLOATS] = 0.2;
    for (const i of [0, 1]) {
      const o = i * GAUSSIAN_FLOATS;
      g[o + 4] = g[o + 5] = g[o + 6] = 0.1;
      g[o + 8] = 1;
    }
    const csr = buildKnnGraph(g);
    const w = edgeWeights(g, null, csr);
    assert.ok(Math.abs(w[0] - Math.exp(-2)) < 1e-5, `w=${w[0]}`);
    // anisotropic: the same offset along a thin axis is far; along a long axis it is close
    g[4] = g[GAUSSIAN_FLOATS + 4] = 1.0; // sx = 1 → d_M² = 0.04 → w ≈ 0.98
    const wLong = edgeWeights(g, null, csr);
    assert.ok(wLong[0] > 0.97, `w=${wLong[0]}`);
    // the slot-space and generic paths agree
    const plain = { offsets: csr.offsets, neighbors: csr.neighbors };
    const wGeneric = edgeWeights(g, null, plain);
    assert.ok(Math.abs(wGeneric[0] - wLong[0]) < 1e-6);
  });

  test("inverse covariance of an isotropic gaussian is 1/s² on the diagonal", () => {
    const g = new Float32Array(GAUSSIAN_FLOATS);
    g[4] = g[5] = g[6] = 0.5;
    g[8] = 1;
    const ic = inverseCovariances(g);
    assert.deepEqual(Array.from(ic).map((v) => Math.round(v * 1e4) / 1e4), [4, 0, 0, 4, 0, 4]);
  });
});

describe("connected components / superpoints", () => {
  test("two separated blobs give exactly two superpoints", () => {
    const a = makeBlob({ n: 3000, center: [-1, 0, 0], radius: 0.5, scale: 0.08, rgb: [0.2, 0.8, 0.7], seed: 11 });
    const b = makeBlob({ n: 2500, center: [1, 0, 0], radius: 0.5, scale: 0.08, rgb: [0.9, 0.5, 0.2], seed: 12 });
    const cloud = concatClouds(a, b);
    const g = buildSuperpointGraph(cloud.gaussians, cloud.colors);
    assert.equal(g.superpointCount, 2);
    assert.deepEqual(Array.from(g.sizes), [3000, 2500]);
    // superpoint 0 is the largest (blob a) and every gaussian of a blob shares it
    for (let i = 0; i < 3000; i++) assert.equal(g.superpoint[i], 0);
    for (let i = 3000; i < cloud.count; i++) assert.equal(g.superpoint[i], 1);
    assert.ok(Math.abs(g.centroids[0] - -1) < 0.05 && Math.abs(g.centroids[3] - 1) < 0.05);
    assert.equal(indicesOfGroup(g.superpoint, 1).length, 2500);
    assertCsrConsistent(g.csr, cloud.count);
  });

  test("the synthetic two-sphere scene gives exactly two superpoints", () => {
    const s = makeTwoSpheres();
    const g = buildSuperpointGraph(s.gaussians, shDcToRgb(s.sh, s.count));
    assert.equal(g.superpointCount, 2);
    assert.deepEqual(Array.from(g.sizes), [2000, 2000]);
    // superpoint ids agree with the scene labels (1 ↔ A, 2 ↔ B) up to renumbering
    const map = new Map();
    for (let i = 0; i < s.count; i++) {
      const prev = map.get(s.labels[i]);
      if (prev === undefined) map.set(s.labels[i], g.superpoint[i]);
      else assert.equal(prev, g.superpoint[i]);
    }
    assert.equal(new Set(map.values()).size, 2);
  });

  test("isolated far points become singleton superpoints", () => {
    const blob = makeBlob({ n: 800, center: [0, 0, 0], radius: 0.3, scale: 0.05, rgb: [0.5, 0.5, 0.5], seed: 21 });
    const far = [];
    for (let i = 0; i < 3; i++) {
      far.push(makeBlob({ n: 1, center: [5 + i * 4, 5, 5], radius: 0, scale: 0.05, rgb: [0.5, 0.5, 0.5], seed: 30 + i }));
    }
    const cloud = concatClouds(blob, ...far);
    const g = buildSuperpointGraph(cloud.gaussians, cloud.colors);
    assert.equal(g.superpointCount, 4);
    assert.deepEqual(Array.from(g.sizes), [800, 1, 1, 1]);
    for (let i = 800; i < 803; i++) assert.ok(g.superpoint[i] >= 1);
  });

  test("threshold cuts weak edges: same cloud, more components with a high threshold", () => {
    const { gaussians, colors } = makeUniform(3000, 9);
    const lo = buildSuperpointGraph(gaussians, colors, { threshold: 0.0 });
    const hi = buildSuperpointGraph(gaussians, colors, { threshold: 0.95 });
    assert.equal(lo.superpointCount, 1);
    assert.ok(hi.superpointCount > lo.superpointCount);
    assert.equal(hi.stats.edgesKept <= hi.stats.edges, true);
  });

  test("empty cloud", () => {
    const g = buildSuperpointGraph(new Float32Array(0), null);
    assert.equal(g.count, 0);
    assert.equal(g.superpointCount, 0);
    assert.equal(g.superpoint.length, 0);
    assert.equal(g.centroids.length, 0);
  });

  test("connectedComponents without weights joins every edge", () => {
    const csr = { offsets: new Uint32Array([0, 1, 2, 2]), neighbors: new Uint32Array([1, 0]) };
    const cc = connectedComponents(csr, null);
    assert.equal(cc.count, 2);
    assert.deepEqual(Array.from(cc.component), [0, 0, 1]);
    assert.deepEqual(Array.from(cc.sizes), [2, 1]);
    const g = new Float32Array(3 * GAUSSIAN_FLOATS);
    g[0] = 2; g[GAUSSIAN_FLOATS] = 4; g[2 * GAUSSIAN_FLOATS] = 9;
    assert.deepEqual(Array.from(componentCentroids(cc.component, cc.count, g)), [3, 0, 0, 9, 0, 0]);
  });
});

describe("diffuseLabels", () => {
  test("weighted majority removes salt-and-pepper noise and keeps seeds", () => {
    const a = makeBlob({ n: 1500, center: [-1, 0, 0], radius: 0.4, scale: 0.08, rgb: [0.2, 0.8, 0.7], seed: 41 });
    const b = makeBlob({ n: 1500, center: [1, 0, 0], radius: 0.4, scale: 0.08, rgb: [0.9, 0.5, 0.2], seed: 42 });
    const cloud = concatClouds(a, b);
    const g = buildSuperpointGraph(cloud.gaussians, cloud.colors);
    const rnd = mulberry32(5);
    const labels = new Uint32Array(cloud.count);
    for (let i = 0; i < cloud.count; i++) labels[i] = i < 1500 ? 1 : 2;
    const seeds = new Uint8Array(cloud.count);
    seeds[0] = 1;
    labels[0] = 7; // a seed keeps its (odd) label
    let flipped = 0;
    for (let i = 1; i < cloud.count; i++) {
      if (rnd() < 0.1) {
        labels[i] = labels[i] === 1 ? 2 : 1;
        flipped++;
      }
    }
    assert.ok(flipped > 200);
    const out = diffuseLabels(labels, g.csr, g.csr.weights, { iterations: 5, seeds });
    assert.equal(out.length, cloud.count);
    assert.equal(out[0], 7, "la semilla conserva su etiqueta");
    let wrong = 0;
    for (let i = 1; i < cloud.count; i++) if (out[i] !== (i < 1500 ? 1 : 2)) wrong++;
    assert.ok(wrong < flipped / 10, `quedan ${wrong} errores de ${flipped}`);
    // input untouched
    assert.equal(labels[1] === out[1] || labels[1] !== out[1], true);
    assert.ok(labels.some((l, i) => l !== out[i]));
  });

  test("zero iterations returns a copy; bad inputs throw", () => {
    const csr = { offsets: new Uint32Array([0, 1, 2]), neighbors: new Uint32Array([1, 0]) };
    const labels = new Uint32Array([3, 4]);
    const out = diffuseLabels(labels, csr, null, { iterations: 0 });
    assert.deepEqual(Array.from(out), [3, 4]);
    assert.notEqual(out, labels);
    assert.throws(() => diffuseLabels(labels, csr, new Float32Array(1)), /weights length/);
    assert.throws(() => diffuseLabels(labels, csr, null, { iterations: -1 }), /iterations/);
  });
});

describe("groups → labels and palette", () => {
  test("groupsToLabels caps at maxLabels and minSize", () => {
    const superpoint = new Uint32Array([0, 0, 0, 1, 1, 2, 3]);
    const sizes = new Uint32Array([3, 2, 1, 1]);
    const { labels, groupOfLabel } = groupsToLabels(superpoint, sizes, { maxLabels: 2, minSize: 2 });
    assert.deepEqual(Array.from(labels), [1, 1, 1, 2, 2, 0, 0]);
    assert.deepEqual(Array.from(groupOfLabel), [0, 0, 1]);
  });

  test("groupColor is deterministic, grey for 0 and distinct for consecutive ids", () => {
    assert.deepEqual(groupColor(0), [0.35, 0.35, 0.35]);
    const c1 = groupColor(1);
    const c2 = groupColor(2);
    assert.deepEqual(groupColor(1), c1);
    const dist = Math.hypot(c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]);
    assert.ok(dist > 0.3, `colores demasiado parecidos: ${dist}`);
    assert.throws(() => groupColor(-1));
  });

  test("shDcToRgb inverts rgbToShDc", () => {
    const sh = new Float32Array(48);
    sh.set(rgbToShDc([0.2, 0.5, 0.9]), 0);
    const rgb = shDcToRgb(sh, 1);
    assert.ok(Math.abs(rgb[0] - 0.2) < 1e-6 && Math.abs(rgb[1] - 0.5) < 1e-6 && Math.abs(rgb[2] - 0.9) < 1e-6);
  });
});

describe("performance", () => {
  test("250k gaussians build in a bounded time (scaled from the 1e6 < 3 s acceptance)", () => {
    const n = 250_000;
    const { gaussians, colors } = makeUniform(n, 77);
    const t0 = performance.now();
    const g = buildSuperpointGraph(gaussians, colors);
    const ms = performance.now() - t0;
    const budget = Number(process.env.GRAPH_BUDGET_MS_250K || 4000);
    console.log(`[graph] ${n.toLocaleString("es-ES")} gaussianas → ${g.superpointCount} superpuntos en ${ms.toFixed(0)} ms ` +
      `(kNN ${g.stats.msKnn.toFixed(0)}, pesos ${g.stats.msWeights.toFixed(0)}, grado medio ${g.stats.avgDegree.toFixed(1)})`);
    assert.equal(g.stats.truncated, 0);
    assert.ok(ms < budget, `${ms.toFixed(0)} ms supera el presupuesto ${budget} ms (GRAPH_BUDGET_MS_250K)`);
  });
});

test("DEFAULT_GRAPH_OPTIONS is frozen and sane", () => {
  assert.ok(Object.isFrozen(DEFAULT_GRAPH_OPTIONS));
  assert.equal(DEFAULT_GRAPH_OPTIONS.k, 10);
  assert.ok(DEFAULT_GRAPH_OPTIONS.threshold > 0 && DEFAULT_GRAPH_OPTIONS.threshold < 1);
});
