/**
 * F4 acceptance (plan §4 "F4 Nombrar"): every instance gets nombre, categoría
 * and confianza from the sidecar (mocked here with page.route, so no API key or
 * network is needed), the isolated per-instance render is what gets sent, the
 * text search highlights and selects the matching instance, and the export
 * carries the names. Also covers the per-instance Imagine card path.
 */
import { test, expect } from "@playwright/test";

const VIEWER_PAGE = "/gaussian_splatting_webgpu/index.html?offscreen=1&scene=synthetic";
const SIDECAR = "http://127.0.0.1:8766";
const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET, POST, OPTIONS" };
/** 1x1 PNG (red) as the mocked Imagine output. */
const TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

const MOCK_NAMES = {
  1: { nombre: "teal sphere", nombre_es: "esfera turquesa", categoria: "decoracion", confianza: 0.93, descripcion_es: "esfera lisa de color turquesa" },
  2: { nombre: "orange sphere", nombre_es: "esfera naranja", categoria: "decoracion", confianza: 0.88, descripcion_es: "esfera lisa de color naranja" },
};

/** Mock the sidecar endpoints the viewer calls; records the /name request bodies. */
async function mockSidecar(page, calls) {
  await page.route(`${SIDECAR}/**`, async (route) => {
    const req = route.request();
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    const url = new URL(req.url());
    if (url.pathname === "/health") {
      return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, xai: false, name_backend: "mock", vision_model: "mock" }) });
    }
    if (url.pathname === "/name") {
      const body = req.postDataJSON();
      calls.push(body);
      const instances = body.instances.map((it, k) => ({ ok: true, id_instancia: it.id, ...(MOCK_NAMES[it.id] || { nombre: `object ${it.id}`, nombre_es: `objeto ${it.id}`, categoria: "otro", confianza: 0.5, descripcion_es: "" }) }));
      return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, backend: body.backend, vision_model: null, instances }) });
    }
    if (url.pathname === "/card") {
      const body = req.postDataJSON();
      calls.push({ card: body.name, pngLength: (body.png_b64 || "").length });
      return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, mime: "image/png", b64: TINY_PNG_B64, url: "", path: "img_output/mock/imagine.png" }) });
    }
    return route.fulfill({ status: 404, headers: CORS, body: "{}" });
  });
}

function pipeConsole(page) {
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning" || /^\[(nombres)\]/.test(msg.text())) console.log(`[browser:${msg.type()}] ${msg.text().slice(0, 300)}`);
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
}

async function openViewer(page) {
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(() => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsNames && !!window.__gsSegment, null, { timeout: 60_000 });
  await page.waitForFunction(() => window.__gsInstances.project([-1, 0, 0]) !== null, null, { timeout: 15_000 });
}

test.beforeEach(async ({ page }) => {
  pipeConsole(page);
});

test("Nombrar: isolated crops go to the sidecar and every instance gets nombre_es, categoría and confianza", async ({ page }) => {
  const calls = [];
  await mockSidecar(page, calls);
  await openViewer(page);

  // The isolated crop of sphere A shows only A: framed, opaque white background, no orange pixels.
  const crop = await page.evaluate(async () => {
    const png = await window.__gsNames.crop(1);
    const img = new Image();
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = `data:image/png;base64,${png}`; });
    const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
    const ctx = c.getContext("2d", { willReadFrequently: true }); ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let teal = 0, orange = 0, white = 0;
    for (let p = 0; p < d.length; p += 4) { const r = d[p], g = d[p + 1], b = d[p + 2]; if (r > 240 && g > 240 && b > 240) white++; else if (b > r + 30) teal++; else if (r > b + 40) orange++; }
    const restored = { isolate: window.__gsRenderer.params.isolateLabel, target: window.__gsCamera.target.map((v) => +v.toFixed(3)) };
    return { width: img.width, height: img.height, teal, orange, white, total: d.length / 4, restored };
  });
  console.log(`[f4] recorte aislado de A: ${crop.width}x${crop.height} · turquesa ${crop.teal} · naranja ${crop.orange} · blanco ${crop.white} · cámara restaurada ${JSON.stringify(crop.restored)}`);
  expect(crop.width).toBe(512);
  expect(crop.teal, "la esfera A aparece en su recorte").toBeGreaterThan(crop.total * 0.05);
  expect(crop.orange, "la esfera B no aparece en el recorte de A").toBe(0);
  expect(crop.white, "fondo blanco opaco").toBeGreaterThan(crop.total * 0.3);
  expect(crop.restored.isolate, "el aislamiento se restaura tras el recorte").toBe(0);
  expect(crop.restored.target, "la cámara vuelve a su objetivo").toEqual([0, 0, 0]);

  const before = await page.evaluate(() => window.__gsNames.entries().map((r) => [r.label, r.name, r.categoria]));
  expect(before).toEqual([[1, "esfera A", ""], [2, "esfera B", ""]]);

  await page.locator("#inst-name").click();
  await page.waitForFunction(() => document.getElementById("inst-name-status").dataset.kind === "ok", null, { timeout: 60_000 });
  const after = await page.evaluate(() => ({
    rows: window.__gsNames.entries().map((r) => ({ label: r.label, name: r.name, nombre: r.nombre, categoria: r.categoria, confianza: r.confianza })),
    status: document.getElementById("inst-name-status").textContent,
    metas: [...document.querySelectorAll("#inst-list .inst-meta")].map((m) => m.textContent),
    names: [...document.querySelectorAll("#inst-list .inst-name")].map((m) => m.textContent),
  }));
  console.log(`[f4] tras nombrar: ${JSON.stringify(after.rows)} · ${after.status}`);
  expect(calls.length, "una llamada a /name con las dos instancias").toBe(1);
  expect(calls[0].backend).toBe("mock");
  expect(calls[0].instances.map((i) => i.id)).toEqual([1, 2]);
  for (const it of calls[0].instances) expect(it.png_b64.length, "cada instancia envía su recorte PNG").toBeGreaterThan(1000);
  expect(after.rows).toEqual([
    { label: 1, name: "esfera turquesa", nombre: "teal sphere", categoria: "decoracion", confianza: 0.93 },
    { label: 2, name: "esfera naranja", nombre: "orange sphere", categoria: "decoracion", confianza: 0.88 },
  ]);
  expect(after.names).toEqual(["esfera turquesa", "esfera naranja"]);
  expect(after.metas[0]).toBe("decoracion · 93 % · teal sphere");
  expect(after.status).toMatch(/^2 instancias nombradas · mock · \d+ ms$/);

  // Text search highlights and Enter selects the best match.
  await page.fill("#inst-search", "naranja");
  const highlighted = await page.evaluate(() => ({
    match: [...document.querySelectorAll("#inst-list .inst-row.match")].map((r) => r.dataset.label),
    dim: [...document.querySelectorAll("#inst-list .inst-row.dim")].map((r) => r.dataset.label),
  }));
  expect(highlighted.match).toEqual(["2"]);
  expect(highlighted.dim).toEqual(["1"]);
  await page.press("#inst-search", "Enter");
  expect(await page.evaluate(() => window.__gsInstances.current?.label), "Intro selecciona la instancia encontrada").toBe(2);
  await expect(page.locator("#inst-status")).toHaveText(/^Seleccionada: instancia 2 \(esfera naranja\)/);
  const byCategory = await page.evaluate(() => window.__gsNames.search("decoracion", false).map((r) => r.label));
  expect(byCategory, "la búsqueda por categoría encuentra ambas").toEqual([1, 2]);
  const bySpanishAccent = await page.evaluate(() => window.__gsNames.search("Turquésa").map((r) => r.label));
  expect(bySpanishAccent, "la búsqueda ignora acentos y mayúsculas").toEqual([1]);
  expect(await page.evaluate(() => window.__gsInstances.current?.label)).toBe(1);
  await page.fill("#inst-search", "");
  expect(await page.evaluate(() => document.querySelectorAll("#inst-list .inst-row.dim").length)).toBe(0);

  // Export carries the names (no lift needed: método "manual").
  const exported = await page.evaluate(() => window.__gsSegment.build().json);
  expect(exported.metodo.mascaras).toBe("manual");
  expect(exported.instancias.map((i) => [i.id_instancia, i.nombre, i.nombre_es, i.categoria, i.confianza])).toEqual([
    [1, "teal sphere", "esfera turquesa", "decoracion", 0.93],
    [2, "orange sphere", "esfera naranja", "decoracion", 0.88],
  ]);

  // Per-instance Imagine card through the mocked /card endpoint, tied to id_instancia.
  const card = await page.evaluate(() => window.__gsNames.card(2));
  expect(card).toEqual({ label: 2, path: "img_output/mock/imagine.png" });
  const fig = page.locator('#sem-cards figure[data-label="2"]');
  await expect(fig.locator("figcaption")).toHaveText("#2 esfera naranja · img_output/mock/imagine.png");
  expect(calls.at(-1).card).toBe("orange sphere");
  expect(calls.at(-1).pngLength).toBeGreaterThan(1000);
});

test("Nombrar without a sidecar reports a clear Spanish error and keeps the names", async ({ page }) => {
  await page.route(`${SIDECAR}/**`, (route) => route.abort("connectionrefused"));
  await openViewer(page);
  const err = await page.evaluate(async () => {
    try {
      await window.__gsNames.name();
      return null;
    } catch (e) {
      return { message: e.message, status: document.getElementById("inst-name-status").textContent, kind: document.getElementById("inst-name-status").dataset.kind, rows: window.__gsNames.entries().map((r) => r.name) };
    }
  });
  expect(err).not.toBeNull();
  expect(err.kind).toBe("err");
  expect(err.status).toMatch(/^Nombrado fallido: .*semantic_sidecar\/launch\.sh/);
  expect(err.rows).toEqual(["esfera A", "esfera B"]);
});
