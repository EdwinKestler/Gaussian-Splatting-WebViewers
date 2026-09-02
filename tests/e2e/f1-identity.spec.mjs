// F1 identity acceptance tests (plan §4, fase F1):
//   - escena sintética de dos esferas: el clic (pick) sobre cada una devuelve su etiqueta;
//   - profundidad frente al valor analítico < 1 %;
//   - ocultar / aislar / transformar instancias se refleja en el pase de ID;
//   - normal en espacio de cámara orientada hacia la cámara.
//
// Everything renders to offscreen GPUTextures through renderer.renderOffscreen()
// / renderer.pick(); no canvas WebGPU context is ever configured, so the suite
// runs under headless SwiftShader (see docs/testing.md).
//
// Browser-side helpers live in tests/e2e/pages/f1-harness.js, loaded by
// tests/e2e/pages/f1-identity.html and driven here through page.evaluate().

import { test, expect } from "@playwright/test";

const HARNESS_PAGE = "/tests/e2e/pages/f1-identity.html";
const SELFTEST_PAGE = "/gaussian_splatting_webgpu/selftest.html";

const LABEL_A = 1;
const LABEL_B = 2;
/** Minimum number of covered pixels expected around each sphere centre (plan F1). */
const MIN_COVERAGE_PX = 200;
/** F1 acceptance criterion: |depth - analytic| / analytic < 1 %. */
const DEPTH_TOLERANCE = 0.01;
/** Camera used for the two-sphere scene: on the +z axis looking at the origin. */
const SPHERES_EYE = [0, 0, 4];

/** Forward browser console output to the test log. */
function pipeConsole(page) {
  page.on("console", (msg) => {
    const where = msg.type() === "error" && msg.location().url ? ` (${msg.location().url})` : "";
    // Keep multi-line dumps (selftest.html prints its whole result) to one short line.
    const text = msg.text().replace(/\s+/g, " ");
    console.log(`[browser:${msg.type()}] ${text.length > 240 ? text.slice(0, 240) + "…" : text}${where}`);
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
}

/** Open the harness page and wait for tests/e2e/pages/f1-harness.js to load. */
async function openHarness(page) {
  await page.goto(HARNESS_PAGE);
  await page.waitForFunction(() => window.__f1Ready === true || !!window.__f1Error, null, { timeout: 30_000 });
  const loadError = await page.evaluate(() => window.__f1Error || null);
  expect(loadError, "f1-harness.js no se pudo importar").toBeNull();
}

/** Create the offscreen renderer inside the page (kept in window.__h). */
async function initHarness(page, testInfo, options = {}) {
  const info = await page.evaluate(async (opts) => {
    const h = await new window.__f1.F1Harness(opts).init();
    window.__h = h;
    return { adapter: h.adapter, floatFormat: h.renderer.floatFormat, width: h.width, height: h.height };
  }, options);
  const summary = `vendor=${info.adapter?.vendor ?? "?"} architecture=${info.adapter?.architecture ?? "?"} float=${info.floatFormat}`;
  testInfo.annotations.push({ type: "webgpu-adapter", description: summary });
  console.log(`[f1] renderer offscreen ${info.width}x${info.height} · ${summary}`);
  return info;
}

/** Fail if the device was lost or produced uncaptured errors. */
async function expectHealthy(page) {
  const health = await page.evaluate(() => window.__h.health());
  expect(health.lost, "el dispositivo WebGPU se perdió durante la prueba").toBeNull();
  expect(health.uncaptured, "errores WebGPU no capturados").toEqual([]);
  return health;
}

/** Load makeTwoSpheres() into the harness and return the projected geometry. */
async function loadTwoSpheres(page) {
  const setup = await page.evaluate((eye) => {
    const f1 = window.__f1;
    const h = window.__h;
    const scene = f1.makeTwoSpheres();
    h.setScene(scene);
    const focal = h.lookFrom(eye);
    const [cA, cB] = scene.centers;
    return {
      count: scene.count,
      centers: scene.centers,
      radius: scene.radius,
      focal,
      width: h.width,
      height: h.height,
      pxA: h.project(cA),
      pxB: h.project(cB),
      radiusPx: (scene.radius * focal) / h.viewDepth(cA),
      frontDepthA: h.frontDepth(cA, scene.radius),
      frontDepthB: h.frontDepth(cB, scene.radius),
    };
  }, SPHERES_EYE);
  expect(setup.pxA, "el centro de la esfera A no se proyecta").not.toBeNull();
  expect(setup.pxB, "el centro de la esfera B no se proyecta").not.toBeNull();
  console.log(
    `[f1] dos esferas: ${setup.count} gaussianas · centros proyectados A=${setup.pxA} B=${setup.pxB} · radio ${setup.radiusPx.toFixed(1)} px`
  );
  return setup;
}

/** ID statistics plus the id/pick result at the two sphere centres. */
async function readIds(page, setup) {
  return page.evaluate(async ({ pxA, pxB }) => {
    const h = window.__h;
    const stats = await h.idStats();
    return {
      stats,
      atA: h.idAt(pxA[0], pxA[1]),
      atB: h.idAt(pxB[0], pxB[1]),
      pickA: await h.pick(pxA[0], pxA[1]),
      pickB: await h.pick(pxB[0], pxB[1]),
    };
  }, setup);
}

const pixelsOf = (stats, label) => stats.byLabel[label]?.pixels ?? 0;

test.beforeEach(async ({ page }) => {
  pipeConsole(page);
});

// ---------------------------------------------------------------- (a) selftest

test("selftest.html reports SELFTEST_OK with pickA=1, pickB=2 and coverage > 200 px", async ({ page }, testInfo) => {
  await page.goto(SELFTEST_PAGE);
  await page.waitForFunction(() => document.title.startsWith("SELFTEST_"), null, { timeout: 60_000 });
  const title = await page.title();
  const r = await page.evaluate(() => window.__selftest);
  console.log(`[f1] ${title} · depthErrA=${r?.depthErrA} · ${r?.ms} ms`);
  if (r?.adapter) {
    testInfo.annotations.push({ type: "webgpu-adapter", description: `vendor=${r.adapter.vendor} architecture=${r.adapter.architecture}` });
  }

  expect(r, "window.__selftest no existe").toBeTruthy();
  expect(r.lost, "selftest: dispositivo WebGPU perdido").toBeNull();
  expect(title, `selftest falló: ${r.reason}`).toMatch(/^SELFTEST_OK /);
  expect(r.ok, `selftest.ok=false: ${r.reason}`).toBe(true);
  expect(r.pickA, "pickA (etiqueta en el centro de la esfera A)").toBe(LABEL_A);
  expect(r.pickB, "pickB (etiqueta en el centro de la esfera B)").toBe(LABEL_B);
  expect(r.pickApiA?.label, "renderer.pick() en el centro de A").toBe(LABEL_A);
  expect(r.pickApiB?.label, "renderer.pick() en el centro de B").toBe(LABEL_B);
  expect(r.covA, "cobertura de color alrededor de A").toBeGreaterThan(MIN_COVERAGE_PX);
  expect(r.covB, "cobertura de color alrededor de B").toBeGreaterThan(MIN_COVERAGE_PX);
  expect(r.depthErrA, "sin profundidad en el centro de A").not.toBeNull();
  expect(r.depthErrA, "error de profundidad de A frente al valor analítico (criterio F1: < 1 %)").toBeLessThan(DEPTH_TOLERANCE);
  expect(r.idPixelsBHidden, "píxeles id de B tras ocultarla").toBe(0);
  expect(r.idPixelsBIsolatedA, "píxeles id de B tras aislar A").toBe(0);
});

// ------------------------------------------------- (b) direct renderer API

test("two spheres: id readback and pick() return each sphere's label; hide/isolate remove sphere B", async ({ page }, testInfo) => {
  await openHarness(page);
  await initHarness(page, testInfo);
  const setup = await loadTwoSpheres(page);
  const half = Math.max(2, Math.floor(setup.radiusPx / 2));

  // Colour coverage around each centre (same box as selftest.html).
  const colour = await page.evaluate(async ({ pxA, pxB, half }) => {
    const h = window.__h;
    return { a: await h.colourAt(pxA[0], pxA[1], half), b: await h.colourAt(pxB[0], pxB[1], half) };
  }, { ...setup, half });
  console.log(`[f1] cobertura color A=${colour.a.covered}/${colour.a.total} B=${colour.b.covered}/${colour.b.total} · rgba A=${colour.a.rgba} B=${colour.b.rgba}`);
  expect(colour.a.covered, "cobertura de color alrededor de A").toBeGreaterThan(MIN_COVERAGE_PX);
  expect(colour.b.covered, "cobertura de color alrededor de B").toBeGreaterThan(MIN_COVERAGE_PX);
  expect(colour.a.rgba[3], "alpha del píxel central de A").toBeGreaterThan(0);
  expect(colour.b.rgba[3], "alpha del píxel central de B").toBeGreaterThan(0);
  // Sphere A is teal (green > red), sphere B orange (red > green).
  expect(colour.a.rgba[1], "la esfera A debe ser verde-azulada").toBeGreaterThan(colour.a.rgba[0]);
  expect(colour.b.rgba[0], "la esfera B debe ser naranja").toBeGreaterThan(colour.b.rgba[1]);

  // ID pass + pick() at the projected centres.
  const ids = await readIds(page, setup);
  console.log(
    `[f1] id: A→${JSON.stringify(ids.atA)} B→${JSON.stringify(ids.atB)} · pick A=${JSON.stringify(ids.pickA)} B=${JSON.stringify(ids.pickB)} · píxeles por etiqueta ${JSON.stringify(Object.fromEntries(Object.entries(ids.stats.byLabel).map(([l, s]) => [l, s.pixels])))}`
  );
  expect(ids.stats.invalid, "ids fuera de [1, N] en el pase de ID").toBe(0);
  expect(ids.atA.label, "etiqueta del pase de ID en el centro de A").toBe(LABEL_A);
  expect(ids.atB.label, "etiqueta del pase de ID en el centro de B").toBe(LABEL_B);
  expect(ids.pickA.label, "renderer.pick() en el centro de A").toBe(LABEL_A);
  expect(ids.pickB.label, "renderer.pick() en el centro de B").toBe(LABEL_B);
  expect(ids.pickA.index, "pick() y la lectura de ID deben dar la misma gaussiana (A)").toBe(ids.atA.index);
  expect(ids.pickB.index, "pick() y la lectura de ID deben dar la misma gaussiana (B)").toBe(ids.atB.index);
  expect(pixelsOf(ids.stats, LABEL_A), "píxeles id de la esfera A").toBeGreaterThan(MIN_COVERAGE_PX);
  expect(pixelsOf(ids.stats, LABEL_B), "píxeles id de la esfera B").toBeGreaterThan(MIN_COVERAGE_PX);
  // A is entirely on the left half of the image, B on the right half.
  const mid = setup.width / 2;
  expect(ids.stats.byLabel[LABEL_A].xRange[1], "píxeles de A en la mitad derecha").toBeLessThan(mid);
  expect(ids.stats.byLabel[LABEL_B].xRange[0], "píxeles de B en la mitad izquierda").toBeGreaterThan(mid);
  expect(pixelsOf(ids.stats, 0), "píxeles id con etiqueta 0 (fondo) en una escena totalmente etiquetada").toBe(0);

  // pick() outside both spheres hits nothing.
  const empty = await page.evaluate(() => window.__h.pick(0, 0));
  expect(empty, "pick() en un píxel vacío").toEqual({ index: -1, label: 0, depth: null });

  // Depth at the centre of each sphere vs the analytic front-surface view depth.
  const depth = await page.evaluate(async ({ pxA, pxB }) => {
    const h = window.__h;
    return { a: await h.depthAt(pxA[0], pxA[1]), b: await h.depthAt(pxB[0], pxB[1]), pickDepthA: await h.pick(pxA[0], pxA[1], { depth: true }) };
  }, setup);
  const errA = Math.abs(depth.a.depth - setup.frontDepthA) / setup.frontDepthA;
  const errB = Math.abs(depth.b.depth - setup.frontDepthB) / setup.frontDepthB;
  console.log(
    `[f1] profundidad A=${depth.a.depth?.toFixed(4)} (analítica ${setup.frontDepthA.toFixed(4)}, error ${(errA * 100).toFixed(2)} %) · B=${depth.b.depth?.toFixed(4)} (analítica ${setup.frontDepthB.toFixed(4)}, error ${(errB * 100).toFixed(2)} %) · formato ${depth.a.format}`
  );
  expect(depth.a.depth, "sin profundidad en el centro de A").not.toBeNull();
  expect(depth.b.depth, "sin profundidad en el centro de B").not.toBeNull();
  expect(errA, "error de profundidad en A (criterio F1: < 1 %)").toBeLessThan(DEPTH_TOLERANCE);
  expect(errB, "error de profundidad en B (criterio F1: < 1 %)").toBeLessThan(DEPTH_TOLERANCE);
  expect(depth.pickDepthA.depth, "pick({depth:true}) debe devolver la misma profundidad que el readback").toBeCloseTo(depth.a.depth, 3);

  // Hide instance 2 → sphere B disappears from the ID pass; A is untouched.
  await page.evaluate((label) => window.__h.renderer.setInstance(label, { visible: false }), LABEL_B);
  const hidden = await readIds(page, setup);
  console.log(`[f1] ocultar B → píxeles id de B ${pixelsOf(hidden.stats, LABEL_B)} · centro B ${JSON.stringify(hidden.atB)} · centro A ${JSON.stringify(hidden.atA)}`);
  expect(pixelsOf(hidden.stats, LABEL_B), "píxeles id de B tras ocultarla").toBe(0);
  expect(hidden.atB, "el centro de B debe quedar vacío tras ocultarla").toEqual({ index: -1, label: 0 });
  expect(hidden.pickB.index, "pick() en el centro de B tras ocultarla").toBe(-1);
  expect(hidden.atA.label, "la esfera A debe seguir visible al ocultar B").toBe(LABEL_A);
  expect(pixelsOf(hidden.stats, LABEL_A), "píxeles id de A al ocultar B").toBe(pixelsOf(ids.stats, LABEL_A));

  // Restore B, isolate instance 1 → again only A is drawn.
  await page.evaluate(({ labelB, labelA }) => {
    const r = window.__h.renderer;
    r.setInstance(labelB, { visible: true });
    r.setParams({ isolateLabel: labelA });
  }, { labelB: LABEL_B, labelA: LABEL_A });
  const isolated = await readIds(page, setup);
  console.log(`[f1] aislar A → píxeles id de B ${pixelsOf(isolated.stats, LABEL_B)} · píxeles id de A ${pixelsOf(isolated.stats, LABEL_A)}`);
  expect(pixelsOf(isolated.stats, LABEL_B), "píxeles id de B tras aislar A").toBe(0);
  expect(isolated.atB, "el centro de B debe quedar vacío tras aislar A").toEqual({ index: -1, label: 0 });
  expect(isolated.pickB.index, "pick() en el centro de B tras aislar A").toBe(-1);
  expect(isolated.atA.label, "la esfera A debe verse al aislarla").toBe(LABEL_A);
  expect(isolated.pickA.label, "pick() en A al aislarla").toBe(LABEL_A);

  // Isolate off → B is back.
  await page.evaluate(() => window.__h.renderer.setParams({ isolateLabel: 0 }));
  const restored = await readIds(page, setup);
  expect(restored.atB.label, "la esfera B debe reaparecer al quitar el aislamiento").toBe(LABEL_B);
  expect(restored.pickB.label, "pick() en B al quitar el aislamiento").toBe(LABEL_B);
  expect(pixelsOf(restored.stats, LABEL_B), "píxeles id de B restaurados").toBe(pixelsOf(ids.stats, LABEL_B));

  await expectHealthy(page);
});

// ------------------------------------------------------ (c) analytic depth

test("depth readback of a single gaussian matches the analytic view distance (< 1 %)", async ({ page }, testInfo) => {
  await openHarness(page);
  await initHarness(page, testInfo);

  // Expected depth is the view-space distance -cam.z of the gaussian centre
  // (not the Euclidean distance: the off-axis case tells the two apart).
  const cases = [
    { name: "origen, cámara a 4", position: [0, 0, 0], eye: [0, 0, 4], expected: 4 },
    { name: "origen, cámara a 2.5", position: [0, 0, 0], eye: [0, 0, 2.5], expected: 2.5 },
    { name: "origen, cámara a 7", position: [0, 0, 0], eye: [0, 0, 7], expected: 7 },
    { name: "fuera de eje (1.2, 0.8, 0), cámara a 4", position: [1.2, 0.8, 0], eye: [0, 0, 4], expected: 4 },
    { name: "desplazada en z (0, 0, -1), cámara a 4", position: [0, 0, -1], eye: [0, 0, 4], expected: 5 },
  ];

  const results = await page.evaluate(async (cases) => {
    const f1 = window.__f1;
    const h = window.__h;
    const out = [];
    for (const c of cases) {
      h.setScene(f1.makeSingleGaussian({ position: c.position, scale: 0.05, opacity: 1 }));
      h.lookFrom(c.eye);
      const px = h.project(c.position);
      const centre = await h.depthAt(px[0], px[1]);
      const corner = await h.depthAt(0, 0);
      out.push({ ...c, px, viewDepth: h.viewDepth(c.position), centre, corner });
    }
    return out;
  }, cases);

  for (const r of results) {
    const err = r.centre.depth == null ? null : Math.abs(r.centre.depth - r.expected) / r.expected;
    console.log(`[f1] profundidad "${r.name}": píxel ${r.px} · medida ${r.centre.depth} · esperada ${r.expected} · error ${err == null ? "n/d" : (err * 100).toFixed(3) + " %"} · alpha ${r.centre.alpha.toFixed(3)}`);
    expect(r.viewDepth, `profundidad de vista analítica (${r.name})`).toBeCloseTo(r.expected, 5);
    expect(r.centre.alpha, `sin cobertura en el centro de la gaussiana (${r.name})`).toBeGreaterThan(0);
    expect(r.centre.depth, `sin profundidad en el centro (${r.name})`).not.toBeNull();
    expect(err, `error de profundidad (${r.name}), criterio F1: < 1 %`).toBeLessThan(DEPTH_TOLERANCE);
    expect(r.corner.alpha, `el píxel (0,0) debería estar vacío (${r.name})`).toBe(0);
    expect(r.corner.depth, `profundidad de fondo debe ser null (${r.name})`).toBeNull();
  }

  await expectHealthy(page);
});

// ----------------------------------------------------------- (d) normals

test("normal readback: smallest-scale axis in view space, flipped to face the camera", async ({ page }, testInfo) => {
  await openHarness(page);
  await initHarness(page, testInfo);

  const s45 = Math.SQRT1_2;
  // Flat disc (smallest scale along z) seen from +z; quaternion = rotation about y.
  const cases = [
    { name: "disco frontal", axis: [0, 1, 0], deg: 0, xformDeg: 0, expected: [0, 0, 1] },
    { name: "disco inclinado 45°", axis: [0, 1, 0], deg: 45, xformDeg: 0, expected: [s45, 0, s45] },
    // Axis (sin 135°, 0, cos 135°) points away from the camera → must be flipped.
    { name: "disco inclinado 135° (eje alejándose, volteado)", axis: [0, 1, 0], deg: 135, xformDeg: 0, expected: [-s45, 0, s45] },
    // Same as the 45° case but the tilt comes from the instance xform (label 0).
    { name: "disco frontal + xform de instancia 45°", axis: [0, 1, 0], deg: 0, xformDeg: 45, expected: [s45, 0, s45] },
  ];

  const results = await page.evaluate(async (cases) => {
    const f1 = window.__f1;
    const h = window.__h;
    const out = [];
    for (const c of cases) {
      const scene = f1.makeSingleGaussian({ position: [0, 0, 0], scale: [0.3, 0.3, 0.01], opacity: 1 });
      scene.gaussians.set(f1.quatAxisAngle(c.axis, c.deg), 8); // [qw, qx, qy, qz]
      h.setScene(scene);
      // Instance 0 (fondo) xform: rotation about y as a column-major mat4.
      const t = (c.xformDeg * Math.PI) / 180;
      const cs = Math.cos(t);
      const sn = Math.sin(t);
      h.renderer.setInstance(0, { xform: [cs, 0, -sn, 0, 0, 1, 0, 0, sn, 0, cs, 0, 0, 0, 0, 1] });
      h.lookFrom([0, 0, 4]);
      const px = h.project([0, 0, 0]);
      const centre = await h.normalAt(px[0], px[1]);
      const corner = await h.normalAt(0, 0);
      h.renderer.resetInstances();
      out.push({ ...c, px, centre, corner });
    }
    return out;
  }, cases);

  for (const r of results) {
    const n = r.centre.normal;
    const dot = n ? n[0] * r.expected[0] + n[1] * r.expected[1] + n[2] * r.expected[2] : null;
    console.log(`[f1] normal "${r.name}": medida ${n ? n.map((v) => v.toFixed(3)) : "n/d"} · esperada ${r.expected.map((v) => v.toFixed(3))} · cos ${dot?.toFixed(4)} · formato ${r.centre.format}`);
    expect(r.centre.alpha, `sin cobertura en el centro del disco (${r.name})`).toBeGreaterThan(0);
    expect(n, `sin normal en el centro (${r.name})`).not.toBeNull();
    expect(Math.hypot(n[0], n[1], n[2]), `la normal debe ser unitaria (${r.name})`).toBeCloseTo(1, 2);
    expect(n[2], `la normal debe mirar hacia la cámara (+z de vista) (${r.name})`).toBeGreaterThan(0);
    // cos(angle) > 0.99 → angle < ~8°.
    expect(dot, `la normal se desvía de la esperada (${r.name})`).toBeGreaterThan(0.99);
    expect(r.corner.alpha, `el píxel (0,0) debería estar vacío (${r.name})`).toBe(0);
    expect(r.corner.normal, `normal de fondo debe ser null (${r.name})`).toBeNull();
  }

  await expectHealthy(page);
});

// ------------------------------------------------- (e) rigid transform

test("setInstance(2, { xform: translation }) moves sphere B's id pixels", async ({ page }, testInfo) => {
  await openHarness(page);
  await initHarness(page, testInfo);
  const setup = await loadTwoSpheres(page);

  const before = await readIds(page, setup);
  expect(before.atB.label, "estado inicial: centro de B").toBe(LABEL_B);
  expect(before.atA.label, "estado inicial: centro de A").toBe(LABEL_A);

  // Move B up by more than its radius so its old centre pixel becomes empty.
  const translation = [0, 0.8, 0];
  expect(translation[1], "la traslación debe superar el radio para vaciar el centro antiguo").toBeGreaterThan(setup.radius);
  const moved = await page.evaluate(async ({ setup, translation, labelB }) => {
    const f1 = window.__f1;
    const h = window.__h;
    const xform = f1.translationMatrix(translation);
    h.renderer.setInstance(labelB, { xform });
    const cB = setup.centers[1];
    const newCentre = [cB[0] + translation[0], cB[1] + translation[1], cB[2] + translation[2]];
    const pxNew = h.project(newCentre);
    const stats = await h.idStats();
    return {
      xform: Array.from(h.renderer.getInstance(labelB).xform),
      pxNew,
      stats,
      atOld: h.idAt(setup.pxB[0], setup.pxB[1]),
      atNew: h.idAt(pxNew[0], pxNew[1]),
      atA: h.idAt(setup.pxA[0], setup.pxA[1]),
      pickOld: await h.pick(setup.pxB[0], setup.pxB[1]),
      pickNew: await h.pick(pxNew[0], pxNew[1]),
    };
  }, { setup, translation, labelB: LABEL_B });

  const centroidBefore = before.stats.byLabel[LABEL_B].centroid;
  const centroidAfter = moved.stats.byLabel[LABEL_B]?.centroid ?? [NaN, NaN];
  const expectedShift = [moved.pxNew[0] - setup.pxB[0], moved.pxNew[1] - setup.pxB[1]];
  const shift = [centroidAfter[0] - centroidBefore[0], centroidAfter[1] - centroidBefore[1]];
  console.log(
    `[f1] traslación ${translation}: centro B ${setup.pxB} → ${moved.pxNew} · centroide id B ${centroidBefore.map((v) => v.toFixed(1))} → ${centroidAfter.map((v) => v.toFixed(1))} (desplazamiento ${shift.map((v) => v.toFixed(1))}, esperado ${expectedShift}) · píxeles B ${pixelsOf(before.stats, LABEL_B)} → ${pixelsOf(moved.stats, LABEL_B)}`
  );

  // The renderer stores the record in a Float32Array, so compare with a tolerance.
  const expectedXform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...translation, 1];
  expect(moved.xform, "getInstance(2).xform debe tener 16 elementos").toHaveLength(16);
  expectedXform.forEach((v, i) => expect(moved.xform[i], `getInstance(2).xform[${i}] debe devolver la traslación`).toBeCloseTo(v, 6));
  expect(moved.stats.invalid, "ids fuera de [1, N] tras la traslación").toBe(0);
  expect(moved.atOld.label, "el centro antiguo de B ya no debe ser la etiqueta 2").not.toBe(LABEL_B);
  expect(moved.atOld, "el centro antiguo de B debe quedar vacío").toEqual({ index: -1, label: 0 });
  expect(moved.pickOld.label, "pick() en el centro antiguo de B").not.toBe(LABEL_B);
  expect(moved.atNew.label, "el centro trasladado de B debe ser la etiqueta 2").toBe(LABEL_B);
  expect(moved.pickNew.label, "pick() en el centro trasladado de B").toBe(LABEL_B);
  expect(moved.atA.label, "la esfera A no debe moverse").toBe(LABEL_A);
  expect(pixelsOf(moved.stats, LABEL_A), "píxeles id de A tras trasladar B").toBe(pixelsOf(before.stats, LABEL_A));
  // A pure translation keeps the projected size roughly (perspective aside) and
  // shifts the id-pixel centroid by the projected displacement.
  const pixelsBefore = pixelsOf(before.stats, LABEL_B);
  const pixelsAfter = pixelsOf(moved.stats, LABEL_B);
  expect(Math.abs(pixelsAfter - pixelsBefore) / pixelsBefore, "los píxeles id de B cambian demasiado con una traslación").toBeLessThan(0.15);
  expect(Math.abs(shift[0] - expectedShift[0]), "desplazamiento x del centroide de B").toBeLessThan(4);
  expect(Math.abs(shift[1] - expectedShift[1]), "desplazamiento y del centroide de B").toBeLessThan(4);

  // Identity again → B returns to its original place.
  const reset = await page.evaluate(async ({ setup, labelB }) => {
    const h = window.__h;
    h.renderer.setInstance(labelB, { xform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
    const stats = await h.idStats();
    return { stats, atB: h.idAt(setup.pxB[0], setup.pxB[1]), pickB: await h.pick(setup.pxB[0], setup.pxB[1]) };
  }, { setup, labelB: LABEL_B });
  expect(reset.atB.label, "B debe volver a su sitio con la identidad").toBe(LABEL_B);
  expect(reset.pickB.label, "pick() en B tras restaurar la identidad").toBe(LABEL_B);
  expect(pixelsOf(reset.stats, LABEL_B), "píxeles id de B tras restaurar la identidad").toBe(pixelsBefore);

  await expectHealthy(page);
});
