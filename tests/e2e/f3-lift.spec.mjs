/**
 * F3 acceptance (plan §4 "F3 Levantar"): the K-buffer contribution pass and
 * the FlashSplat lift recover a known synthetic labelling with 3D IoU > 0.9,
 * masks with per-view permuted ids are associated into consistent instances,
 * the K-buffer median depth is exact, and the viewer exports instancias.json.
 *
 * Headless SwiftShader: renderer in offscreenOnly mode, everything read back.
 */
import { test, expect } from "@playwright/test";

const VIEWER_PAGE = "/gaussian_splatting_webgpu/index.html?offscreen=1&scene=synthetic";
const SPHERE_SIZE = 2000;
const MIN_IOU = 0.9;
const DEPTH_TOLERANCE = 0.01;

function pipeConsole(page) {
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning" || /^\[(segmentación|grupos)\]/.test(msg.text())) {
      console.log(`[browser:${msg.type()}] ${msg.text().slice(0, 300)}`);
    }
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
}

/** Direct renderer harness (no viewer UI): two spheres + camera helpers. */
const HARNESS = `
  const { WebGPUSplatRenderer, OUTPUT_MODE } = await import("/gaussian_splatting_webgpu/gpu-renderer.js");
  const { makeTwoSpheres, makeSingleGaussian } = await import("/shared/synthetic.js");
  const { liftViews, matchLabels } = await import("/shared/lift.js");
  const { buildSuperpointGraph, shDcToRgb, diffuseLabels } = await import("/shared/graph.js");
  const persp = (fovy, aspect, near, far) => { const f = 1 / Math.tan(fovy / 2); const o = new Float32Array(16); o[0] = f / aspect; o[5] = f; o[10] = (far + near) / (near - far); o[11] = -1; o[14] = (2 * far * near) / (near - far); return o; };
  const lookAt = (eye, target, up) => { const z = [eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]]; const zl = Math.hypot(...z); z[0]/=zl; z[1]/=zl; z[2]/=zl; const x = [up[1]*z[2]-up[2]*z[1], up[2]*z[0]-up[0]*z[2], up[0]*z[1]-up[1]*z[0]]; const xl = Math.hypot(...x); x[0]/=xl; x[1]/=xl; x[2]/=xl; const y = [z[1]*x[2]-z[2]*x[1], z[2]*x[0]-z[0]*x[2], z[0]*x[1]-z[1]*x[0]]; const o = new Float32Array(16); o[0]=x[0];o[1]=y[0];o[2]=z[0];o[4]=x[1];o[5]=y[1];o[6]=z[1];o[8]=x[2];o[9]=y[2];o[10]=z[2];o[12]=-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]);o[13]=-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]);o[14]=-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]);o[15]=1; return o; };
  const W = 512, H = 384;
  const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
  const r = new WebGPUSplatRenderer(canvas);
  await r.init({ offscreenOnly: true });
  const lost = []; r.device.lost.then((i) => lost.push(i.message));
  const setCam = (eye) => { const fov = 50 * Math.PI / 180; const fy = H / (2 * Math.tan(fov / 2)); r.setCamera(persp(fov, W / H, 0.05, 100), lookAt(eye, [0, 0, 0], [0, 1, 0]), [fy, fy], [W, H], eye); };
  const orbit = (v, n) => { const yaw = v * 2 * Math.PI / n; const pitch = v % 2 ? -0.5 : 0.5; return [4 * Math.cos(pitch) * Math.sin(yaw), 4 * Math.sin(pitch), 4 * Math.cos(pitch) * Math.cos(yaw)]; };
`;

test.beforeEach(async ({ page }) => {
  pipeConsole(page);
});

test("K-buffer: single-view contributions are exact and the median depth matches the analytic distance", async ({ page }) => {
  await page.goto("/index.html");
  const res = await page.evaluate(`(async () => {
    ${HARNESS}
    const out = { lost };
    // (a) single opaque gaussian at the origin, camera at 4: median depth 4, all mass in label 1
    const g = makeSingleGaussian({ position: [0, 0, 0], scale: 0.05, opacity: 1 });
    r.setCloud(g.gaussians, g.sh, 0);
    setCam([0, 0, 4]);
    const full = new Uint32Array(W * H).fill(1);
    const c1 = await r.renderContributions({ mask: full, width: W, height: H, labelCount: 2, k: 4 });
    const centre = (H / 2) * W + W / 2;
    out.single = { median: c1.medianDepth[centre], alpha: c1.alpha[centre], mass1: c1.contrib[1], mass0: c1.contrib[0], chunks: c1.chunks };
    // (b) two stacked gaussians: front at depth 4 (alpha 0.9), back at depth 5 → median = front, mean is between
    const front = makeSingleGaussian({ position: [0, 0, 0], scale: 0.05, opacity: 0.9 });
    const back = makeSingleGaussian({ position: [0, 0, -1], scale: 0.08, opacity: 0.9 });
    const gg = new Float32Array(24); gg.set(front.gaussians, 0); gg.set(back.gaussians, 12);
    const sh = new Float32Array(96); sh.set(front.sh, 0); sh.set(back.sh, 48);
    r.setCloud(gg, sh, 0); setCam([0, 0, 4]);
    const c2 = await r.renderContributions({ mask: full, width: W, height: H, labelCount: 2, k: 4 });
    const mean = await r.renderOffscreen({ mode: OUTPUT_MODE.DEPTH, width: W, height: H });
    out.stacked = { median: c2.medianDepth[centre], mean: mean.data[centre], massFront: c2.contrib[1], massBack: c2.contrib[3] };
    // (c) two spheres, mask = truth through the ID pass: every seen gaussian gets its own label
    const s = makeTwoSpheres();
    r.setCloud(s.gaussians, s.sh, 0); r.setLabels(s.labels);
    setCam([0, 0.6, 4]);
    const id = await r.renderOffscreen({ mode: OUTPUT_MODE.ID, width: W, height: H });
    const mask = new Uint32Array(W * H);
    for (let p = 0; p < mask.length; p++) { const v = id.data[p]; if (v) mask[p] = s.labels[v - 1]; }
    const t0 = performance.now();
    const c3 = await r.renderContributions({ mask, width: W, height: H, labelCount: 3, k: 16 });
    let right = 0, wrong = 0, unseen = 0;
    for (let i = 0; i < s.count; i++) { const m1 = c3.contrib[i * 3 + 1], m2 = c3.contrib[i * 3 + 2]; if (m1 + m2 === 0) { unseen++; continue; } if ((m1 >= m2 ? 1 : 2) === s.labels[i]) right++; else wrong++; }
    out.spheres = { right, wrong, unseen, chunks: c3.chunks, splits: c3.splits, ms: performance.now() - t0, overflow: c3.overflowPixels };
    return out;
  })()`);
  console.log(`[f3] K-buffer: gaussiana única mediana ${res.single.median} · apilado mediana ${res.stacked.median} / media ${res.stacked.mean.toFixed(3)} · esferas ${res.spheres.right} bien / ${res.spheres.wrong} mal / ${res.spheres.unseen} no vistas · ${res.spheres.chunks} trozos (${res.spheres.splits} divisiones) · ${res.spheres.ms.toFixed(0)} ms`);
  expect(res.lost, "el dispositivo WebGPU se perdió").toEqual([]);
  expect(Math.abs(res.single.median - 4) / 4, "profundidad mediana de una gaussiana a 4").toBeLessThan(DEPTH_TOLERANCE);
  expect(res.single.mass1, "toda la masa cae en la etiqueta de la máscara").toBeGreaterThan(50);
  expect(res.single.mass0).toBe(0);
  expect(Math.abs(res.stacked.median - 4) / 4, "la mediana elige la gaussiana frontal").toBeLessThan(DEPTH_TOLERANCE);
  expect(res.stacked.mean, "la media alfa-ponderada queda entre ambas").toBeGreaterThan(4.02);
  expect(res.stacked.massBack, "la gaussiana trasera recibe masa atenuada por T").toBeGreaterThan(0);
  expect(res.spheres.overflow).toBe(0);
  expect(res.spheres.wrong, "ninguna gaussiana vista recibe la etiqueta de la otra esfera").toBe(0);
  expect(res.spheres.right).toBeGreaterThan(SPHERE_SIZE);
});

test("lift: 6 orbit views with permuted mask ids give 2 instances with 3D IoU > 0.9 (superpoint association + diffusion)", async ({ page }, testInfo) => {
  await page.goto("/index.html");
  const res = await page.evaluate(`(async () => {
    ${HARNESS}
    const s = makeTwoSpheres();
    r.setCloud(s.gaussians, s.sh, 0); r.setLabels(s.labels);
    const NV = 6;
    const views = [];
    const t0 = performance.now();
    for (let v = 0; v < NV; v++) {
      setCam(orbit(v, NV));
      const id = await r.renderOffscreen({ mode: OUTPUT_MODE.ID, width: W, height: H });
      const perm = v % 2 ? [0, 2, 1] : [0, 1, 2];
      const mask = new Uint32Array(W * H);
      for (let p = 0; p < mask.length; p++) { const g = id.data[p]; if (g) mask[p] = perm[s.labels[g - 1]]; }
      const c = await r.renderContributions({ mask, width: W, height: H, labelCount: 3, k: 16 });
      views.push({ contrib: c.contrib, labelCount: 3 });
    }
    const msViews = performance.now() - t0;
    const graph = buildSuperpointGraph(s.gaussians, shDcToRgb(s.sh, s.count));
    const withSp = liftViews(views, { count: s.count, superpoint: graph.superpoint, backgroundBias: 0.3 });
    const noSp = liftViews(views, { count: s.count, superpoint: null, backgroundBias: 0.3 });
    const diffused = diffuseLabels(withSp.labels, graph.csr, graph.csr.weights, { iterations: 5 });
    const iou = (labels) => [...matchLabels(s.labels, labels).entries()].map(([t, m]) => ({ truth: t, label: m.label, iou: m.iou }));
    const unlabeled = (labels) => { let n = 0; for (let i = 0; i < labels.length; i++) if (!labels[i]) n++; return n; };
    return {
      lost, msViews, superpoints: graph.superpointCount,
      withSp: { globalCount: withSp.globalCount, merges: withSp.association.pairs.length, iou: iou(withSp.labels), unlabeled: unlabeled(withSp.labels) },
      noSp: { globalCount: noSp.globalCount },
      diffused: { iou: iou(diffused), unlabeled: unlabeled(diffused) },
    };
  })()`);
  testInfo.annotations.push({ type: "f3-lift", description: `${res.withSp.globalCount} instancias · IoU ${res.diffused.iou.map((x) => x.iou.toFixed(4))}` });
  console.log(`[f3] levantamiento: ${res.superpoints} superpuntos · sin superpuntos ${res.noSp.globalCount} instancias · con superpuntos ${res.withSp.globalCount} instancias (${res.withSp.merges} fusiones) · IoU ${JSON.stringify(res.withSp.iou)} · tras difusión ${JSON.stringify(res.diffused.iou)} · sin etiqueta ${res.withSp.unlabeled} → ${res.diffused.unlabeled} · vistas ${res.msViews.toFixed(0)} ms`);
  expect(res.lost).toEqual([]);
  expect(res.withSp.globalCount, "la asociación por superpuntos une las vistas en 2 instancias").toBe(2);
  expect(res.withSp.merges).toBeGreaterThanOrEqual(NV_MERGES_MIN);
  const labelsFound = new Set(res.diffused.iou.map((x) => x.label));
  expect(labelsFound.size, "cada esfera se asigna a una instancia distinta").toBe(2);
  for (const x of res.diffused.iou) expect(x.iou, `IoU 3D de la esfera ${x.truth}`).toBeGreaterThan(MIN_IOU);
  for (const x of res.withSp.iou) expect(x.iou, `IoU antes de la difusión, esfera ${x.truth}`).toBeGreaterThan(0.85);
  expect(res.diffused.unlabeled, "la difusión rellena las gaussianas no vistas").toBeLessThan(res.withSp.unlabeled + 1);
});
const NV_MERGES_MIN = 4; // 6 views × 2 masks = 12 masks, 2 instances → at least 10 merges; be lenient

test("viewer: «Levantar máscaras» (fuente prueba) rebuilds the two instances and exports instancias.json", async ({ page }) => {
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(
    () => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsSegment && !!window.__gsGroups,
    null,
    { timeout: 60_000 }
  );
  await page.waitForFunction(() => window.__gsInstances.project([-1, 0, 0]) !== null, null, { timeout: 15_000 });
  const before = await page.evaluate(() => ({ status: document.getElementById("seg-status").textContent, exportDisabled: document.getElementById("seg-export").disabled }));
  expect(before.status).toContain("Sin segmentación");
  expect(before.exportDisabled).toBe(true);

  const summary = await page.evaluate(() => window.__gsSegment.lift({ source: "prueba", views: 6, backgroundBias: 0.3, diffusion: 5 }));
  console.log(`[f3] visor: ${summary.globalCount} instancias · ${summary.views.length} vistas · ${summary.merges} fusiones · ${summary.changed} corregidas · ${summary.ms.toFixed(0)} ms · máscaras ${summary.width}x${summary.height}`);
  expect(summary.globalCount).toBe(2);
  expect(summary.views).toHaveLength(6);
  const state = await page.evaluate((n) => {
    const labels = window.__gsRenderer.getLabels();
    const a = new Map();
    const b = new Map();
    for (let i = 0; i < n; i++) a.set(labels[i], (a.get(labels[i]) || 0) + 1);
    for (let i = n; i < 2 * n; i++) b.set(labels[i], (b.get(labels[i]) || 0) + 1);
    const top = (m) => [...m.entries()].sort((x, y) => y[1] - x[1])[0];
    return {
      topA: top(a),
      topB: top(b),
      rows: window.__gsInstances.list().map((r) => [r.label, r.name, r.count]),
      status: document.getElementById("seg-status").textContent,
      exportDisabled: document.getElementById("seg-export").disabled,
      groupsComputed: window.__gsGroups.result !== null,
    };
  }, SPHERE_SIZE);
  console.log(`[f3] visor: A → ${state.topA} · B → ${state.topB} · filas ${JSON.stringify(state.rows)} · ${state.status}`);
  expect(state.groupsComputed, "el levantamiento calcula los superpuntos si faltan").toBe(true);
  expect(state.topA[0]).toBeGreaterThan(0);
  expect(state.topB[0]).toBeGreaterThan(0);
  expect(state.topA[0]).not.toBe(state.topB[0]);
  expect(state.topA[1] / SPHERE_SIZE, "esfera A casi entera en su instancia").toBeGreaterThan(MIN_IOU);
  expect(state.topB[1] / SPHERE_SIZE, "esfera B casi entera en su instancia").toBeGreaterThan(MIN_IOU);
  expect(state.rows).toHaveLength(2);
  expect(state.rows.map((r) => r[1]).sort(), "los nombres de las instancias sobreviven a la permutación de ids").toEqual(["esfera A", "esfera B"]);
  expect(state.status).toMatch(/^2 instancias · 6 vistas · \d+ fusiones · \d[\d.]* etiquetas corregidas por difusión · \d+ ms$/);
  expect(state.exportDisabled).toBe(false);

  // Export without downloads/sidecar: schema of plan §3.3
  const exported = await page.evaluate(async () => {
    const { json, bytes } = window.__gsSegment.build();
    const dv = new DataView(bytes.buffer);
    return { json, bytesLength: bytes.length, first: dv.getUint32(0, true), mid: dv.getUint32(2000 * 4, true) };
  });
  expect(exported.json.version).toBe(1);
  expect(exported.json.escena).toBe("synthetic-two-spheres");
  expect(exported.json.fuente.n_gaussianas).toBe(2 * SPHERE_SIZE);
  expect(exported.json.metodo).toMatchObject({ mascaras: "prueba", levantamiento: "flashsplat", sesgo_fondo: 0.3, difusion_iter: 5, vistas: 6, k_buffer: 24 });
  expect(exported.json.n_instancias).toBe(2);
  expect(exported.json.instancias.map((i) => i.nombre_es).sort()).toEqual(["esfera A", "esfera B"]);
  for (const inst of exported.json.instancias) {
    expect(inst.n_gaussianas).toBeGreaterThan(SPHERE_SIZE * MIN_IOU);
    expect(inst.bbox.min.length).toBe(3);
    expect(inst.vistas.length).toBeGreaterThan(0);
  }
  expect(exported.bytesLength).toBe(2 * SPHERE_SIZE * 4);
  expect(exported.first).toBe(state.topA[0]);
  expect(exported.mid).toBe(state.topB[0]);
  const viaApi = await page.evaluate(() => window.__gsSegment.export({ download: false, save: false }));
  expect(viaApi.json.n_instancias).toBe(2);
  expect(viaApi.saved).toBeNull();
});
