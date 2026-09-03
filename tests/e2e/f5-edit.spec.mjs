/**
 * F5 acceptance (plan §4 "F5 Editar/Exportar"): selection tools on the ID
 * buffer turn gaussians into instances, an instance can be moved and the move
 * shows in the render, exporting one instance and reloading it shows only that
 * object (with its instance_id restored), SPZ comes out of the vendored
 * GaussForge, and replaying ops.jsonl on a fresh scene reproduces the same
 * state (labels + transforms fingerprint). The sidecar is mocked with
 * page.route, so no server is needed.
 */
import { test, expect } from "@playwright/test";

const VIEWER_PAGE = "/gaussian_splatting_webgpu/index.html?offscreen=1&scene=synthetic";
const SIDECAR = "http://127.0.0.1:8766";
const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET, POST, OPTIONS" };

async function mockSidecar(page, calls) {
  await page.route(`${SIDECAR}/**`, async (route) => {
    const req = route.request();
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    const url = new URL(req.url());
    const body = req.postDataJSON();
    if (url.pathname === "/exportaciones") {
      calls.push({ path: url.pathname, escena: body.escena, id_instancia: body.id_instancia, formato: body.formato, bytes: Buffer.from(body.bytes_b64, "base64").length, metadatos: body.metadatos, ops: body.ops_jsonl });
      const stem = body.id_instancia != null ? `instancia-${body.id_instancia}` : "escena";
      return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, carpeta: `artifacts/exportaciones/${body.escena}`, archivo: `artifacts/exportaciones/${body.escena}/${stem}.${body.formato}`, bytes: 1, metadatos: null, ops: null }) });
    }
    if (url.pathname === "/segmentaciones") {
      calls.push({ path: url.pathname, ops: body.ops_jsonl, base: Buffer.from(body.etiquetas_base_b64 || "", "base64").length, etiquetas: Buffer.from(body.etiquetas_b64 || "", "base64").length });
      return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, carpeta: "artifacts/segmentaciones/mock", ops: "artifacts/segmentaciones/mock/ops.jsonl" }) });
    }
    return route.fulfill({ status: 404, headers: CORS, body: "{}" });
  });
}

async function openViewer(page) {
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(() => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsEdit && !!window.__gsLoad, null, { timeout: 60_000 });
  await page.waitForFunction(() => window.__gsInstances.project([-1, 0, 0]) !== null, null, { timeout: 15_000 });
}

test.beforeEach(async ({ page }) => {
  page.on("console", (msg) => {
    if (msg.type() === "error" || /^\[(edición)\]/.test(msg.text())) console.log(`[browser:${msg.type()}] ${msg.text().slice(0, 300)}`);
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
});

test("Editar: selección → instancia, mover, duplicar/fusionar, deshacer, exportar y recargar, reproducir ops.jsonl", async ({ page }) => {
  const calls = [];
  await mockSidecar(page, calls);
  await openViewer(page);
  await expect(page.locator("#edit-panel h2")).toHaveText("Edición");

  // Sphere selection around the picked gaussian of sphere A → new instance 3.
  const sel = await page.evaluate(async () => {
    const p = window.__gsInstances.project([-1, 0, 0]);
    const hit = await window.__gsRenderer.pick(p[0] * devicePixelRatio, p[1] * devicePixelRatio);
    const n = await window.__gsEdit.select({ tool: "esfera", index: hit.index, radius: 0.3 });
    const status = document.getElementById("edit-sel-status").textContent;
    const painted = window.__gsRenderer.getLabels().filter((l) => l === 4095).length;
    const label = window.__gsEdit.commit("new");
    return { hitLabel: hit.label, n, status, painted, label, rows: window.__gsInstances.list().map((r) => [r.label, r.count]), restored: window.__gsRenderer.getLabels().filter((l) => l === 4095).length };
  });
  console.log(`[f5] esfera: ${JSON.stringify(sel)}`);
  expect(sel.hitLabel).toBe(1);
  expect(sel.n).toBeGreaterThan(50);
  expect(sel.painted, "la selección se pinta con la etiqueta auxiliar").toBe(sel.n);
  expect(sel.status).toMatch(/gaussianas seleccionadas$/);
  expect(sel.label).toBe(3);
  expect(sel.restored, "al confirmar no queda etiqueta auxiliar").toBe(0);
  expect(sel.rows).toEqual([[1, 2000 - sel.n], [2, 2000], [3, sel.n]]);

  // Rectangle selection on sphere B in "quitar" mode sends those gaussians to fondo.
  const rect = await page.evaluate(async () => {
    const p = window.__gsInstances.project([1, 0, 0]);
    const n = await window.__gsEdit.select({ tool: "rect", x0: p[0] - 30, y0: p[1] - 30, x1: p[0] + 30, y1: p[1] + 30 });
    window.__gsEdit.commit("bg");
    return { n, rows: window.__gsInstances.list().map((r) => [r.label, r.count]) };
  });
  expect(rect.n).toBeGreaterThan(20);
  expect(rect.rows[1]).toEqual([2, 2000 - rect.n]);

  // Move sphere B by +1.5 z: the ID pass now finds label 2 at the new spot and fondo at the old one.
  const moved = await page.evaluate(async () => {
    window.__gsEdit.transform(2, { translate: [0, 0, 1.5] });
    const at = async (w) => { const p = window.__gsInstances.project(w); const h = await window.__gsRenderer.pick(p[0] * devicePixelRatio, p[1] * devicePixelRatio); return h.label; };
    return { atNew: await at([1, 0.35, 1.5]), atOld: await at([1, 0.35, 0]), xform: window.__gsEdit.session().xforms[2].slice(12, 15) };
  });
  console.log(`[f5] mover: ${JSON.stringify(moved)}`);
  moved.xform.forEach((v, k) => expect(v).toBeCloseTo([0, 0, 1.5][k], 5));
  expect(moved.atNew).toBe(2);
  expect(moved.atOld).toBe(0);

  // Duplicate, delete + undo (visible again), merge the copy back, rename.
  const dup = await page.evaluate(() => {
    const nueva = window.__gsEdit.duplicate(2);
    const count = window.__gsRenderer.count;
    window.__gsEdit.remove(nueva);
    const hidden = !window.__gsRenderer.getInstance(nueva).visible;
    window.__gsEdit.undo();
    const visibleAgain = window.__gsRenderer.getInstance(nueva).visible;
    window.__gsEdit.merge(nueva, 2);
    window.__gsEdit.rename(2, "esfera naranja");
    return { nueva, count, hidden, visibleAgain, rows: window.__gsInstances.list().map((r) => [r.label, r.name, r.count]), ops: window.__gsEdit.ops().map((o) => o.op), undoDisabled: document.getElementById("edit-undo").disabled };
  });
  console.log(`[f5] duplicar/borrar/deshacer/fusionar: ${JSON.stringify(dup)}`);
  expect(dup.nueva).toBe(4);
  expect(dup.count).toBe(4000 + 2000 - rect.n);
  expect(dup.hidden).toBe(true);
  expect(dup.visibleAgain).toBe(true);
  expect(dup.rows.find((r) => r[0] === 2)).toEqual([2, "esfera naranja", 2 * (2000 - rect.n)]);
  expect(dup.rows.some((r) => r[0] === 4)).toBe(false);
  expect(dup.ops).toEqual(["asignar", "asignar", "transformar", "duplicar", "fusionar", "renombrar"]);
  expect(dup.undoDisabled).toBe(false);

  // Export instance 2 as PLY (instance_id + xform baked) and the scene as SPZ through GaussForge.
  const exported = await page.evaluate(async () => {
    const ply = await window.__gsEdit.export({ scope: "instancia", label: 2, format: "ply", download: false, save: true });
    window.__ply = ply.data;
    const spz = await window.__gsEdit.export({ scope: "escena", format: "spz", download: false, save: true });
    window.__spz = spz.data;
    const { toGaussianCloud, readPlyColumns } = await import("/shared/splat-io.js");
    const back = toGaussianCloud(ply.data.buffer.slice(0), "x.ply");
    let zMean = 0;
    for (let i = 0; i < back.count; i++) zMean += back.gaussians[i * 12 + 2];
    const ids = readPlyColumns(ply.data.buffer.slice(0), ["instance_id"]).instance_id;
    return { ply: { name: ply.name, count: ply.count, saved: ply.saved.archivo, zMean: zMean / back.count, ids: [...new Set(ids)], meta: ply.metadatos.instancias[0] }, spz: { name: spz.name, count: spz.count, bytes: spz.bytes, magic: [spz.data[0], spz.data[1]], saved: spz.saved.archivo }, status: document.getElementById("edit-status").textContent };
  });
  console.log(`[f5] exportación: ${JSON.stringify(exported)}`);
  expect(exported.ply.name).toBe("synthetic-two-spheres_instancia-2.ply");
  expect(exported.ply.count).toBe(2 * (2000 - rect.n));
  expect(Math.abs(exported.ply.zMean - 1.5), "la traslación queda horneada en el PLY").toBeLessThan(0.05);
  expect(exported.ply.ids).toEqual([2]);
  expect(exported.ply.meta.nombre_es).toBe("esfera naranja");
  expect(exported.ply.saved).toBe("artifacts/exportaciones/synthetic-two-spheres/instancia-2.ply");
  expect(exported.spz.magic, "SPZ es gzip").toEqual([31, 139]);
  expect(exported.spz.count).toBe(dup.count);
  expect(exported.spz.bytes).toBeLessThan(exported.spz.count * 8);
  expect(exported.status).toMatch(/^synthetic-two-spheres_escena\.spz: .* guardado en /);
  const exportCalls = calls.filter((c) => c.path === "/exportaciones");
  expect(exportCalls.map((c) => [c.id_instancia, c.formato])).toEqual([[2, "ply"], [null, "spz"]]);
  expect(exportCalls[0].ops.split("\n").filter(Boolean).length).toBe(6);

  // ops.jsonl saved next to the base labels it replays over.
  const saved = await page.evaluate(() => window.__gsEdit.saveOps());
  expect(saved.ok).toBe(true);
  const segCall = calls.find((c) => c.path === "/segmentaciones");
  expect(segCall.base).toBe(4000 * 4);
  expect(segCall.etiquetas).toBe(dup.count * 4);

  // Reload the exported instance: only that object, with its instance restored from instance_id.
  const fingerprint = await page.evaluate(() => window.__gsEdit.fingerprint());
  const jsonl = await page.evaluate(() => window.__gsEdit.jsonl());
  const reloaded = await page.evaluate(async () => {
    await window.__gsLoad.buffer(window.__ply.buffer.slice(0), "instancia-2.ply");
    return { viewer: window.__gsViewer, rows: window.__gsInstances.list().map((r) => [r.label, r.count]), labels: [...new Set(window.__gsRenderer.getLabels())] };
  });
  console.log(`[f5] recarga: ${JSON.stringify(reloaded)}`);
  expect(reloaded.viewer.decoder).toBe("gaussforge");
  expect(reloaded.viewer.labelSource).toBe("instance_id");
  expect(reloaded.rows).toEqual([[2, exported.ply.count]]);
  expect(reloaded.labels).toEqual([2]);

  // The SPZ written by GaussForge loads back with every gaussian of the visible scene.
  const spzBack = await page.evaluate(async () => {
    await window.__gsLoad.buffer(window.__spz.buffer.slice(0), "escena.spz");
    return { format: window.__gsViewer.format, decoder: window.__gsViewer.decoder, count: window.__gsViewer.count };
  });
  expect(spzBack).toEqual({ format: "spz", decoder: "gaussforge", count: dup.count });

  // Replay the same ops.jsonl on a fresh synthetic scene → identical fingerprint.
  const replayed = await page.evaluate((text) => {
    window.__gsLoad.synthetic();
    const n = window.__gsEdit.replay(text);
    return { n, fingerprint: window.__gsEdit.fingerprint(), rows: window.__gsInstances.list().map((r) => [r.label, r.name, r.count]) };
  }, jsonl);
  console.log(`[f5] reproducción: ${JSON.stringify(replayed)} (esperado ${fingerprint})`);
  expect(replayed.n).toBe(6);
  expect(replayed.fingerprint).toBe(fingerprint);
  expect(replayed.rows).toEqual(dup.rows);
});

test("Editar: HUD buttons apply a move and undo it; Ctrl+Z undoes; errors are Spanish", async ({ page }) => {
  await page.route(`${SIDECAR}/**`, (route) => route.abort("connectionrefused"));
  await openViewer(page);
  await page.click("#edit-apply");
  await expect(page.locator("#edit-status")).toHaveText("selecciona una instancia");
  await page.evaluate(() => window.__gsInstances.select(1));
  await page.fill("#edit-tx", "0.5");
  await page.click("#edit-apply");
  await expect(page.locator("#edit-status")).toHaveText("Instancia 1 transformada");
  const near = (v, e) => v.forEach((x, k) => expect(x).toBeCloseTo(e[k], 5));
  near(await page.evaluate(() => window.__gsEdit.session().xforms[1].slice(12, 15)), [0.5, 0, 0]);
  expect(await page.inputValue("#edit-tx")).toBe("0");
  await page.keyboard.press("Control+z");
  await expect(page.locator("#edit-status")).toHaveText("Deshecho: transformar");
  expect(await page.evaluate(() => window.__gsEdit.session().xforms)).toEqual({});
  await page.click("#edit-redo");
  near(await page.evaluate(() => window.__gsEdit.session().xforms[1].slice(12, 15)), [0.5, 0, 0]);
  // Export without a sidecar still produces the file (download only).
  const r = await page.evaluate(() => window.__gsEdit.export({ scope: "instancia", label: 1, format: "splat", download: false, save: true }));
  expect(r.count).toBe(2000);
  expect(r.bytes).toBe(2000 * 32);
  expect(r.saved).toBeNull();
  await expect(page.locator("#edit-status")).toHaveText(/· descarga local$/);
});
