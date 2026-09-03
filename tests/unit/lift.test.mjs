/**
 * Unit tests for shared/lift.js (plan F3: FlashSplat assignment, cross-view
 * association, export schema). Run: npm test. Contribution matrices are
 * simulated on the CPU from known labels; the GPU pass is covered by
 * tests/e2e/f3-lift.spec.mjs.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_LIFT_OPTIONS,
  assignLabels,
  associateMasks,
  buildInstancesJson,
  containment,
  labelIou,
  labelsFromBytes,
  labelsToBytes,
  liftViews,
  maskHistogram,
  matchLabels,
  mergeContributions,
  weightedJaccard,
} from "../../shared/lift.js";
import { mulberry32 } from "../../shared/synthetic.js";

/**
 * Simulate a view's contribution matrix: each gaussian carrying truth label t
 * gets mass in the view-local column perm[t] (visible fraction `seen`), some
 * spill into fondo, and `noise` fraction of its mass onto a random other label.
 */
function simulateView(truth, perm, localCount, { seen = 0.7, spill = 0.2, noise = 0.05, seed = 1 } = {}) {
  const rnd = mulberry32(seed);
  const n = truth.length;
  const contrib = new Float32Array(n * localCount);
  for (let i = 0; i < n; i++) {
    const t = truth[i];
    const row = i * localCount;
    if (rnd() > seen) continue; // not visible from this view
    const mass = 5 + rnd() * 20;
    const local = t ? perm[t] : 0;
    contrib[row + local] += mass * (1 - spill - noise);
    contrib[row] += mass * spill;
    const other = 1 + Math.floor(rnd() * (localCount - 1));
    contrib[row + other] += mass * noise;
  }
  return contrib;
}

function twoBlobTruth(n = 3000) {
  const truth = new Uint32Array(n);
  for (let i = 0; i < n; i++) truth[i] = i < n / 2 ? 1 : 2;
  return truth;
}

/** superpoints: 10 per blob, contiguous ranges */
function blobSuperpoints(n = 3000, per = 10) {
  const sp = new Uint32Array(n);
  const half = n / 2;
  for (let i = 0; i < n; i++) sp[i] = i < half ? Math.floor((i / half) * per) : per + Math.floor(((i - half) / half) * per);
  return sp;
}

describe("assignLabels (FlashSplat argmax)", () => {
  test("picks the heaviest foreground column and respects the background bias", () => {
    const count = 4;
    const L = 3;
    const c = new Float32Array([
      0, 10, 2, // → 1
      0, 2, 10, // → 2
      9, 10, 0, // 10 > (1 + 0.3) · 9 = 11.7? no → fondo
      0, 0, 0, // unseen → 0
    ]);
    assert.deepEqual(Array.from(assignLabels(c, count, L)), [1, 2, 0, 0]);
    assert.deepEqual(Array.from(assignLabels(c, count, L, { backgroundBias: -0.5 })), [1, 2, 1, 0]);
    assert.deepEqual(Array.from(assignLabels(c, count, L, { backgroundBias: 0, minMass: 20 })), [0, 0, 0, 0]);
  });

  test("validates inputs", () => {
    assert.throws(() => assignLabels(new Float32Array(5), 2, 3), /count·labelCount/);
    assert.throws(() => assignLabels(new Float32Array(6), 2, 3, { backgroundBias: -2 }), /backgroundBias/);
  });
});

describe("histograms and overlap", () => {
  test("maskHistogram keys by superpoint or index and weightedJaccard is symmetric", () => {
    const labels = new Uint32Array([1, 1, 2, 1, 0]);
    const sp = new Uint32Array([0, 0, 1, 1, 1]);
    const h = maskHistogram(labels, 1, sp);
    assert.equal(h.total, 3);
    assert.deepEqual([...h.hist.entries()], [[0, 2], [1, 1]]);
    const byIndex = maskHistogram(labels, 1);
    assert.deepEqual([...byIndex.hist.keys()], [0, 1, 3]);
    const a = new Map([[0, 2], [1, 1]]);
    const b = new Map([[0, 1], [2, 3]]);
    assert.ok(Math.abs(weightedJaccard(a, b) - 1 / 6) < 1e-9);
    assert.equal(weightedJaccard(a, b), weightedJaccard(b, a));
    assert.equal(weightedJaccard(new Map(), new Map()), 0);
  });
});

describe("associateMasks", () => {
  test("merges the same object across views despite permuted local ids", () => {
    const truth = twoBlobTruth(2000);
    const sp = blobSuperpoints(2000);
    // view 0: A=1, B=2 · view 1: A=2, B=1 · view 2: only B visible as 1
    const v0 = new Uint32Array(truth);
    const v1 = new Uint32Array(truth.map((t) => (t === 1 ? 2 : t === 2 ? 1 : 0)));
    const v2 = new Uint32Array(truth.map((t) => (t === 2 ? 1 : 0)));
    const res = associateMasks(
      [
        { labels: v0, labelCount: 3 },
        { labels: v1, labelCount: 3 },
        { labels: v2, labelCount: 2 },
      ],
      { superpoint: sp }
    );
    assert.equal(res.globalCount, 2);
    // A (1 in v0, 2 in v1) → one global; B (2 in v0, 1 in v1, 1 in v2) → the other
    assert.equal(res.globalOf[0][1], res.globalOf[1][2]);
    assert.equal(res.globalOf[0][2], res.globalOf[1][1]);
    assert.equal(res.globalOf[0][2], res.globalOf[2][1]);
    assert.notEqual(res.globalOf[0][1], res.globalOf[0][2]);
    assert.equal(res.pairs.length, 3, "tres fusiones: A(v1), B(v1), B(v2)"); // first view creates both instances
    assert.ok(res.pairs.every((p) => p.overlap >= 0.5));
    assert.deepEqual(res.members.map((m) => m.length).sort(), [2, 3]);
  });

  test("partial views still associate when the overlap is above the threshold; tiny masks are dropped", () => {
    const truth = twoBlobTruth(2000);
    const sp = blobSuperpoints(2000);
    const v0 = new Uint32Array(truth);
    const v1 = new Uint32Array(2000);
    for (let i = 0; i < 700; i++) v1[i] = 1; // 70 % of A only
    v1[1999] = 2; // a 1-gaussian mask → dropped (minGaussians)
    const res = associateMasks([{ labels: v0, labelCount: 3 }, { labels: v1, labelCount: 3 }], { superpoint: sp });
    assert.equal(res.globalCount, 2);
    assert.equal(res.globalOf[1][1], res.globalOf[0][1]);
    assert.equal(res.globalOf[1][2], 0, "máscara de 1 gaussiana ignorada");
  });

  test("works without superpoints (histogram over gaussian indices)", () => {
    const truth = twoBlobTruth(400);
    const res = associateMasks([{ labels: new Uint32Array(truth), labelCount: 3 }, { labels: new Uint32Array(truth), labelCount: 3 }]);
    assert.equal(res.globalCount, 2);
  });

  test("containment merges two half views of one object; jaccard mode would not", () => {
    // view 0 lifts the left half of blob A, view 1 the right half, both mapped to superpoints
    const n = 2000;
    const sp = blobSuperpoints(n, 10); // 10 superpoints per blob (contiguous ranges of 100)
    const v0 = new Uint32Array(n);
    const v1 = new Uint32Array(n);
    for (let i = 0; i < 600; i++) v0[i] = 1; // superpoints 0..5
    for (let i = 400; i < 1000; i++) v1[i] = 1; // superpoints 4..9 → shared 4,5 (200 of 600 = 0.33)
    const jac = associateMasks([{ labels: v0, labelCount: 2 }, { labels: v1, labelCount: 2 }], { superpoint: sp, mode: "jaccard", iouThreshold: 0.3 });
    assert.equal(jac.globalCount, 2, "jaccard 200/1000 = 0.2 no fusiona");
    const con = associateMasks([{ labels: v0, labelCount: 2 }, { labels: v1, labelCount: 2 }], { superpoint: sp, iouThreshold: 0.3 });
    assert.equal(con.globalCount, 1, "contención 200/600 = 0.33 fusiona");
    assert.ok(Math.abs(con.pairs[0].overlap - 1 / 3) < 1e-9);
    assert.equal(containment(new Map([[1, 2]]), 2, new Map([[1, 1], [2, 5]]), 6), 0.5);
  });
});

describe("liftViews end to end (simulated contributions)", () => {
  test("four views with permuted ids and noise recover both blobs with IoU > 0.9", () => {
    const n = 3000;
    const truth = twoBlobTruth(n);
    const sp = blobSuperpoints(n);
    const perms = [
      { 1: 1, 2: 2 },
      { 1: 2, 2: 1 },
      { 1: 3, 2: 1 }, // a view with a spurious third label
      { 1: 2, 2: 3 },
    ];
    const views = perms.map((perm, v) => ({
      contrib: simulateView(truth, perm, 4, { seed: 10 + v }),
      labelCount: 4,
      names: v === 0 ? ["", "silla", "mesa"] : [],
    }));
    const res = liftViews(views, { count: n, superpoint: sp });
    assert.equal(res.globalCount, 2);
    const match = matchLabels(truth, res.labels);
    for (const t of [1, 2]) {
      const m = match.get(t);
      assert.ok(m.iou > 0.9, `IoU de la instancia ${t}: ${m.iou.toFixed(3)}`);
    }
    assert.notEqual(match.get(1).label, match.get(2).label);
    assert.deepEqual(res.names.slice(1).sort(), ["mesa", "silla"]);
    assert.equal(res.perView.length, 4);
    assert.equal(res.contrib.length, n * 3);
  });

  test("mergeContributions sums mapped columns and keeps fondo", () => {
    const views = [
      { contrib: new Float32Array([1, 2, 3, 4, 5, 6]), labelCount: 3 },
      { contrib: new Float32Array([1, 1, 1, 1]), labelCount: 2 },
    ];
    const globalOf = [new Uint32Array([0, 2, 1]), new Uint32Array([0, 1])];
    const merged = mergeContributions(views, globalOf, 2, 2);
    // gaussian 0: fondo 1+1, g1 ← v0 col2 (3) + v1 col1 (1), g2 ← v0 col1 (2)
    assert.deepEqual(Array.from(merged), [2, 4, 2, 5, 7, 5]);
  });

  test("rejects empty input", () => {
    assert.throws(() => liftViews([], { count: 0 }), /at least one view/);
  });
});

describe("export schema", () => {
  test("buildInstancesJson follows the plan schema in Spanish and labelsToBytes round-trips", () => {
    const labels = new Uint32Array([0, 1, 1, 2, 0]);
    const gaussians = new Float32Array(5 * 12);
    for (let i = 0; i < 5; i++) gaussians[i * 12] = i;
    const json = buildInstancesJson({
      escena: "model.splat",
      fecha: "2026-09-02T00:00:00Z",
      fuente: { formato: "splat", sh_grado: 0 },
      metodo: { mascaras: "prueba", sesgo_fondo: 0.3, umbral_iou: 0.5, difusion_iter: 5, k_buffer: 12 },
      labels,
      gaussians,
      names: ["", { nombre: "chair", nombre_es: "silla", categoria: "mobiliario", confianza: 0.9 }],
      colors: ["", [1, 0, 0]],
      views: [{ indice: 0, instancias: [1, 2] }, { indice: 1, instancias: [1] }],
    });
    assert.equal(json.version, 1);
    assert.equal(json.escena, "model.splat");
    assert.equal(json.fuente.n_gaussianas, 5);
    assert.equal(json.metodo.levantamiento, "flashsplat");
    assert.equal(json.n_instancias, 2);
    assert.deepEqual(json.instancias[0], {
      id_instancia: 1,
      nombre: "chair",
      nombre_es: "silla",
      categoria: "mobiliario",
      confianza: 0.9,
      n_gaussianas: 2,
      bbox: { min: [1, 0, 0], max: [2, 0, 0] },
      color: [1, 0, 0],
      vistas: [0, 1],
      embedding_clip: null,
      malla: null,
    });
    assert.equal(json.instancias[1].nombre_es, "objeto 2");
    assert.deepEqual(json.instancias[1].vistas, [0]);
    assert.equal(json.embeddings, null, "sin CLIP no hay bloque de embeddings");
    assert.equal(json.instancias[0].embedding_clip, null);
    const bytes = labelsToBytes(labels);
    assert.equal(bytes.length, 20);
    assert.deepEqual(Array.from(labelsFromBytes(bytes)), Array.from(labels));
    assert.throws(() => labelsFromBytes(new Uint8Array(3)), /multiple of 4/);
  });

  test("buildInstancesJson carries CLIP embeddings per instance (F4 optional)", () => {
    const labels = new Uint32Array([1, 1, 2]);
    const json = buildInstancesJson({
      escena: "s.splat",
      fecha: "2026-09-03T00:00:00Z",
      fuente: { formato: "splat", sh_grado: 0 },
      metodo: { mascaras: "sam2-navegador" },
      labels,
      embeddings: { modelo: "Xenova/clip-vit-base-patch32", dimension: 3, vectors: { 1: new Float32Array([0.123456, -0.5, 0.86]) } },
    });
    assert.deepEqual(json.embeddings, { modelo: "Xenova/clip-vit-base-patch32", dimension: 3 });
    assert.deepEqual(json.instancias[0].embedding_clip, [0.1235, -0.5, 0.86], "redondeado a 1e-4 y como array JSON");
    assert.equal(json.instancias[1].embedding_clip, null, "instancia sin embedding");
    assert.equal(json.metodo.mascaras, "sam2-navegador");
  });

  test("buildInstancesJson carries the mesh path (F6)", () => {
    const json = buildInstancesJson({ escena: "s", fecha: "f", fuente: {}, metodo: {}, labels: new Uint32Array([1, 2]), names: ["", { nombre_es: "a", malla: "artifacts/mallas/s/1.glb" }] });
    assert.equal(json.instancias[0].malla, "artifacts/mallas/s/1.glb");
    assert.equal(json.instancias[1].malla, null);
  });

  test("labelIou and matchLabels", () => {
    const a = new Uint32Array([1, 1, 2, 0]);
    const b = new Uint32Array([2, 2, 1, 0]);
    assert.equal(labelIou(a, b, 1, 2), 1);
    assert.equal(labelIou(a, b, 1, 1), 0);
    const m = matchLabels(a, b);
    assert.deepEqual(m.get(1), { label: 2, iou: 1 });
    assert.deepEqual(m.get(2), { label: 1, iou: 1 });
  });
});

test("DEFAULT_LIFT_OPTIONS is frozen", () => {
  assert.ok(Object.isFrozen(DEFAULT_LIFT_OPTIONS));
  assert.equal(DEFAULT_LIFT_OPTIONS.iouThreshold, 0.5);
});
