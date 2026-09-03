/**
 * F2 acceptance (plan §4, "F2 Grafo"): the superpoint graph built in
 * shared/graph-worker.js separates the two synthetic spheres into exactly two
 * groups, the «Grupos» colour mode paints them differently, a click with that
 * view active promotes the group under the cursor to an F1 instance, and the
 * label diffusion leaves clean labels untouched.
 *
 * Headless SwiftShader: the viewer runs with ?offscreen=1 (canvas never
 * configured); colours are read back through renderer.renderOffscreen().
 */
import { test, expect } from "@playwright/test";

/** Viewer without instance labels so that promotion creates them. */
const VIEWER_PAGE = "/gaussian_splatting_webgpu/index.html?offscreen=1&scene=synthetic&labels=0";
const SPHERE_A = [-1, 0, 0];
const SPHERE_B = [1, 0, 0];
/** Two-sphere scene: 2000 gaussians per sphere (shared/synthetic.js makeTwoSpheres defaults). */
const SPHERE_SIZE = 2000;
/** Minimum RGB distance (0..255) between the two group colours. */
const MIN_COLOR_DISTANCE = 60;

function pipeConsole(page) {
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning" || msg.text().startsWith("[grupos]")) {
      console.log(`[browser:${msg.type()}] ${msg.text().slice(0, 300)}`);
    }
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
}

async function openViewer(page) {
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(
    () => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsGroups && !!window.__gsInstances,
    null,
    { timeout: 60_000 }
  );
  await page.waitForFunction(() => window.__gsInstances.project([-1, 0, 0]) !== null, null, { timeout: 15_000 });
}

/** Wait for the frame loop to push the HUD params (colour mode) into the renderer. */
const nextFrames = (page, n = 2) =>
  page.evaluate((n) => new Promise((resolve) => {
    const step = (k) => (k <= 0 ? resolve() : requestAnimationFrame(() => step(k - 1)));
    step(n);
  }), n);

/** Colour + alpha of the offscreen colour render at the projection of world point p (device pixels). */
async function colourAtWorld(page, p) {
  return page.evaluate(async (p) => {
    const r = window.__gsRenderer;
    const css = window.__gsInstances.project(p);
    const canvas = document.getElementById("gpu-canvas");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.width;
    const h = canvas.height;
    const x = Math.min(w - 1, Math.max(0, Math.round(css[0] * dpr)));
    const y = Math.min(h - 1, Math.max(0, Math.round(css[1] * dpr)));
    const out = await r.renderOffscreen({ mode: 0, width: w, height: h });
    const o = (y * w + x) * 4;
    return { x, y, rgba: Array.from(out.data.subarray(o, o + 4)), colorMode: r.params.colorMode };
  }, p);
}

async function clickWorldPoint(page, p) {
  const target = await page.evaluate((p) => {
    const px = window.__gsInstances.project(p);
    if (!px) return null;
    const rect = document.getElementById("gpu-canvas").getBoundingClientRect();
    return { px, x: rect.left + px[0], y: rect.top + px[1] };
  }, p);
  expect(target, `el punto ${p} no se proyecta en el lienzo`).not.toBeNull();
  const before = await page.evaluate(() => document.getElementById("inst-status").textContent);
  await page.mouse.click(target.x, target.y);
  await page.waitForFunction(
    (prev) => document.getElementById("inst-status").textContent !== prev,
    before,
    { timeout: 15_000 }
  );
  return page.evaluate(() => ({
    current: window.__gsInstances.current,
    status: document.getElementById("inst-status").textContent,
    rows: window.__gsInstances.list(),
  }));
}

test.beforeEach(async ({ page }) => {
  pipeConsole(page);
});

test("Grupos: the worker graph splits the two spheres into exactly 2 superpoints", async ({ page }, testInfo) => {
  await openViewer(page);
  const before = await page.evaluate(() => ({
    status: document.getElementById("grp-status").textContent,
    result: window.__gsGroups.result,
    diffuseDisabled: document.getElementById("grp-diffuse").disabled,
  }));
  expect(before.result, "sin grafo antes de calcular").toBeNull();
  expect(before.status).toContain("Sin grupos");
  expect(before.diffuseDisabled, "Difundir etiquetas debe estar deshabilitado sin grafo").toBe(true);

  const t0 = Date.now();
  await page.locator("#grp-compute").click();
  await page.waitForFunction(() => window.__gsGroups.result !== null, null, { timeout: 60_000 });
  const ms = Date.now() - t0;
  const r = await page.evaluate(() => window.__gsGroups.result);
  testInfo.annotations.push({ type: "f2-graph", description: `${r.superpointCount} grupos en ${ms} ms (worker ${Math.round(r.stats.msTotal)} ms)` });
  console.log(`[f2] grafo: ${r.superpointCount} superpuntos · tamaños ${r.sizes.slice(0, 4)} · k ${r.k} · celda ${r.cellSize.toFixed(4)} · grado ${r.stats.avgDegree.toFixed(2)} · ${ms} ms`);
  expect(r.count).toBe(2 * SPHERE_SIZE);
  expect(r.superpointCount, "dos esferas separadas → exactamente 2 grupos").toBe(2);
  expect(r.sizes.slice(0, 2)).toEqual([SPHERE_SIZE, SPHERE_SIZE]);
  expect(r.stats.truncated, "ningún punto sin k vecinos en las esferas").toBe(0);

  const hud = await page.evaluate(() => ({
    status: document.getElementById("grp-status").textContent,
    view: window.__gsGroups.view,
    colorMode: document.getElementById("color-mode").value,
    diffuseDisabled: document.getElementById("grp-diffuse").disabled,
    viewActive: document.getElementById("grp-view").classList.contains("active"),
  }));
  expect(hud.status, "estado del panel Grupos").toMatch(/^2 grupos · mayor 2\.?000 · mediana 2\.?000 gaussianas · grado medio 10\.\d · \d+ ms$/);
  expect(hud.view, "calcular activa la vista Grupos").toBe(true);
  expect(hud.colorMode).toBe("4");
  expect(hud.viewActive).toBe(true);
  expect(hud.diffuseDisabled).toBe(false);

  // Every gaussian of a sphere shares one group id and the two ids differ.
  const groups = await page.evaluate((n) => {
    const g = window.__gsRenderer.getGroups();
    const a = new Set();
    const b = new Set();
    for (let i = 0; i < n; i++) a.add(g[i]);
    for (let i = n; i < 2 * n; i++) b.add(g[i]);
    return { a: [...a], b: [...b], groupOf0: window.__gsGroups.groupOf(0), superpointOf0: window.__gsGroups.superpointOf(0) };
  }, SPHERE_SIZE);
  expect(groups.a.length, "esfera A con un único grupo").toBe(1);
  expect(groups.b.length, "esfera B con un único grupo").toBe(1);
  expect(groups.a[0]).not.toBe(groups.b[0]);
  expect(groups.a[0]).toBeGreaterThan(0);
  expect(groups.groupOf0).toBe(groups.superpointOf0 + 1);
});

test("Grupos view colours the two spheres differently and a click promotes the group to an instance", async ({ page }) => {
  await openViewer(page);
  const result = await page.evaluate(() => window.__gsGroups.compute());
  expect(result.superpointCount).toBe(2);
  await nextFrames(page);

  const cA = await colourAtWorld(page, SPHERE_A);
  const cB = await colourAtWorld(page, SPHERE_B);
  console.log(`[f2] vista Grupos (colorMode ${cA.colorMode}): A ${cA.rgba} en ${cA.x},${cA.y} · B ${cB.rgba} en ${cB.x},${cB.y}`);
  expect(cA.colorMode, "el renderer debe estar en modo de color Grupos").toBe(4);
  expect(cA.rgba[3], "cobertura en el centro de A").toBeGreaterThan(0);
  expect(cB.rgba[3], "cobertura en el centro de B").toBeGreaterThan(0);
  const dist = Math.hypot(cA.rgba[0] - cB.rgba[0], cA.rgba[1] - cB.rgba[1], cA.rgba[2] - cB.rgba[2]);
  expect(dist, "los dos grupos deben pintarse con colores distintos").toBeGreaterThan(MIN_COLOR_DISTANCE);

  // The palette colours match shared/graph.js groupColor() (up to shading by luma).
  const palette = await page.evaluate((n) => {
    const gA = window.__gsGroups.groupOf(0);
    const gB = window.__gsGroups.groupOf(n);
    return { gA, gB, colA: window.__gsGroups.groupColor(gA), colB: window.__gsGroups.groupColor(gB) };
  }, SPHERE_SIZE);
  const hueOrder = (rgb) => [0, 1, 2].sort((i, j) => rgb[j] - rgb[i]).join("");
  expect(hueOrder(cA.rgba.slice(0, 3)), "orden de canales de A coincide con la paleta").toBe(hueOrder(palette.colA));
  expect(hueOrder(cB.rgba.slice(0, 3)), "orden de canales de B coincide con la paleta").toBe(hueOrder(palette.colB));

  // No instances yet (labels=0); a click on sphere A with the Grupos view active creates one.
  const rows0 = await page.evaluate(() => window.__gsInstances.list());
  expect(rows0, "sin instancias antes del clic").toEqual([]);
  const a = await clickWorldPoint(page, SPHERE_A);
  console.log(`[f2] clic en A → ${a.status} · filas ${JSON.stringify(a.rows.map((r) => [r.label, r.name, r.count]))}`);
  expect(a.current, "el clic debe seleccionar la instancia creada").not.toBeNull();
  expect(a.current.label).toBe(1);
  expect(a.current.name).toBe(`grupo ${palette.gA}`);
  expect(a.status).toMatch(/^Seleccionada: instancia 1 \(grupo \d+\) · gaussiana \d+$/);
  expect(a.rows).toHaveLength(1);
  expect(a.rows[0]).toMatchObject({ label: 1, count: SPHERE_SIZE, selected: true });
  const labelsA = await page.evaluate((n) => {
    const l = window.__gsRenderer.getLabels();
    let ones = 0;
    let others = 0;
    for (let i = 0; i < l.length; i++) {
      if (l[i] === 1) ones++;
      else if (l[i] !== 0) others++;
    }
    return { ones, others, sphereBLabel: l[n] };
  }, SPHERE_SIZE);
  expect(labelsA.ones, "toda la esfera A recibe la etiqueta 1").toBe(SPHERE_SIZE);
  expect(labelsA.others).toBe(0);
  expect(labelsA.sphereBLabel, "la esfera B sigue siendo fondo").toBe(0);

  // Clicking B creates instance 2; clicking A again re-selects instance 1 (no duplicate).
  const b = await clickWorldPoint(page, SPHERE_B);
  expect(b.current?.label).toBe(2);
  expect(b.rows.map((r) => [r.label, r.count])).toEqual([[1, SPHERE_SIZE], [2, SPHERE_SIZE]]);
  const again = await clickWorldPoint(page, SPHERE_A);
  expect(again.current?.label).toBe(1);
  expect(again.rows).toHaveLength(2);
  expect(await page.evaluate(() => window.__gsGroups.promote(1)), "promote() de un grupo ya promovido devuelve su etiqueta").toBe(
    await page.evaluate((n) => window.__gsRenderer.getLabels()[0], SPHERE_SIZE)
  );

  // Vista Grupos off → the Color selector goes back to the previous mode; pick() reports groups regardless.
  await page.locator("#grp-view").click();
  const off = await page.evaluate(() => ({ view: window.__gsGroups.view, colorMode: document.getElementById("color-mode").value }));
  expect(off.view).toBe(false);
  expect(off.colorMode).toBe("0");
  const pick = await page.evaluate(async (p) => {
    const css = window.__gsInstances.project(p);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    return window.__gsRenderer.pick(css[0] * dpr, css[1] * dpr);
  }, SPHERE_B);
  expect(pick.label).toBe(2);
  expect(pick.group).toBe(palette.gB);
});

test("Difundir etiquetas keeps clean instance labels and repairs flipped ones", async ({ page }) => {
  await openViewer(page);
  await page.evaluate(() => window.__gsGroups.compute());
  await page.evaluate(() => {
    window.__gsGroups.promote(window.__gsGroups.groupOf(0));
    window.__gsGroups.promote(window.__gsGroups.groupOf(2000));
  });
  const clean = await page.evaluate(() => window.__gsGroups.diffuse(3));
  expect(clean, "etiquetas limpias: la difusión no cambia nada").toBe(0);

  // Flip 5 % of sphere A to label 2 and let the graph majority repair them.
  const flipped = await page.evaluate((n) => {
    const l = window.__gsRenderer.getLabels();
    const idx = [];
    for (let i = 0; i < n; i += 20) idx.push(i);
    window.__gsRenderer.setLabel(idx, 2);
    return idx.length;
  }, SPHERE_SIZE);
  const changed = await page.evaluate(() => window.__gsGroups.diffuse(5));
  const after = await page.evaluate((n) => {
    const l = window.__gsRenderer.getLabels();
    let wrongA = 0;
    for (let i = 0; i < n; i++) if (l[i] !== 1) wrongA++;
    return { wrongA, rows: window.__gsInstances.list().map((r) => [r.label, r.count]), status: document.getElementById("status").textContent };
  }, SPHERE_SIZE);
  console.log(`[f2] difusión: ${flipped} etiquetas volteadas → ${changed} cambiadas · errores restantes en A ${after.wrongA} · ${after.status}`);
  expect(changed, "la difusión debe corregir etiquetas").toBeGreaterThan(0);
  expect(after.wrongA, "casi toda la esfera A vuelve a la etiqueta 1").toBeLessThan(flipped / 5);
  expect(after.rows, "el panel recuenta las gaussianas por instancia").toEqual([[1, SPHERE_SIZE - after.wrongA], [2, SPHERE_SIZE + after.wrongA]]);
  expect(after.status).toMatch(/^Difusión de etiquetas: \d[\d.]* gaussianas cambiadas \(5 iteraciones\)$/);
});
