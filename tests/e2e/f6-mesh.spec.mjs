/**
 * F6 acceptance (plan §4 "F6 Malla", milestone H4): "Malla" on an instance
 * or the complete visible scene produces a GLB or print-scaled 3MF. Sphere A of the synthetic scene is orbited (depth + colour),
 * fused into a TSDF in the worker and extracted with surface nets; the mesh is
 * a closed surface whose mean radius matches the sphere within the documented
 * margin, the GLB is a valid glTF 2.0 container, the sidecar (mocked) receives
 * it for artifacts/mallas/, and instancias.json carries the `malla` path.
 */
import { test, expect } from "@playwright/test";

const VIEWER_PAGE = "/gaussian_splatting_webgpu/index.html?offscreen=1&scene=synthetic";
const SIDECAR = "http://127.0.0.1:8766";
const CORS = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Allow-Methods": "GET, POST, OPTIONS" };
/** Documented margin: alpha-weighted mean depth over-estimates the sphere radius by the splat extent (≈ 7 % on the synthetic sphere). */
const RADIUS_TOLERANCE = 0.12;

test.setTimeout(180_000);

test.beforeEach(async ({ page }) => {
  page.on("console", (msg) => {
    if (msg.type() === "error" || /^\[malla\]/.test(msg.text()) && !/vista \d+\//.test(msg.text())) console.log(`[browser:${msg.type()}] ${msg.text().slice(0, 300)}`);
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
});

test("Malla: sphere A → GLB with the right radius, saved through /mallas and referenced by instancias.json", async ({ page }) => {
  const calls = [];
  await page.route(`${SIDECAR}/**`, async (route) => {
    const req = route.request();
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    const url = new URL(req.url());
    if (url.pathname !== "/mallas") return route.fulfill({ status: 404, headers: CORS, body: "{}" });
    const body = req.postDataJSON();
    const glb = Buffer.from(body.glb_b64, "base64");
    const jsonLength = glb.readUInt32LE(12);
    const extras = JSON.parse(glb.subarray(20, 20 + jsonLength).toString("utf8").trim()).extras;
    calls.push({ escena: body.escena, ambito: body.ambito, id_instancia: body.id_instancia, magic: glb.subarray(0, 4).toString("ascii"), bytes: glb.length, metadatos: body.metadatos, extras });
    const stem = body.id_instancia == null ? "escena" : body.id_instancia;
    return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, carpeta: `artifacts/mallas/${body.escena}`, malla: `artifacts/mallas/${body.escena}/${stem}.glb`, bytes: glb.length, metadatos: null }) });
  });
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(() => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsMesh, null, { timeout: 60_000 });
  await page.waitForFunction(() => window.__gsInstances.project([-1, 0, 0]) !== null, null, { timeout: 15_000 });
  await expect(page.locator("#mesh-panel h2")).toHaveText("Malla");
  expect(await page.locator('#inst-list button[data-act="mesh"]').count(), "botón Malla por instancia").toBe(2);

  const r = await page.evaluate(async () => {
    const r = await window.__gsMesh.build(1, { views: 12, resolution: 48, edge: 160, download: false, save: true, returnMesh: true });
    const { decodeGlbHeader } = await import("/shared/glb.js");
    const h = decodeGlbHeader(r.glb);
    const prim = h.json.meshes[0].primitives[0];
    return {
      name: r.name, stats: r.stats, saved: r.saved, bytes: r.bytes, aviso: r.metadatos.aviso, metodo: r.metadatos.metodo,
      glb: { version: h.version, length: h.length, attributes: Object.keys(prim.attributes).sort(), indexCount: h.json.accessors[prim.indices].count, extras: h.json.extras },
      status: document.getElementById("mesh-status").textContent,
      restored: { isolate: window.__gsRenderer.params.isolateLabel, target: window.__gsCamera.target.map((v) => +v.toFixed(3)) },
      malla: window.__gsNames.entries().find((e) => e.label === 1) && window.__gsSegment.build().json.instancias[0].malla,
    };
  });
  console.log(`[f6] ${JSON.stringify({ ...r, stats: { ...r.stats, bbox: undefined } })}`);
  expect(r.name).toBe("synthetic-two-spheres_instancia-1.glb");
  expect(r.stats.vertices).toBeGreaterThan(500);
  expect(r.stats.components).toBe(1);
  expect(Math.abs(r.stats.meanRadius - 0.5) / 0.5, "radio medio de la esfera A (0,5)").toBeLessThan(RADIUS_TOLERANCE);
  expect(r.stats.maxRadius / r.stats.minRadius, "superficie casi esférica").toBeLessThan(1.25);
  for (let a = 0; a < 3; a++) expect(Math.abs(r.stats.centroid[a] - [-1, 0, 0][a])).toBeLessThan(0.03);
  expect(r.metodo.extraccion).toBe("surface-nets");
  expect(r.metodo.vistas).toBe(12);
  expect(r.aviso).toMatch(/3DGS vainilla/);
  expect(r.glb.version).toBe(2);
  expect(r.glb.length).toBe(r.bytes);
  expect(r.glb.attributes).toEqual(["COLOR_0", "NORMAL", "POSITION"]);
  expect(r.glb.indexCount).toBe(r.stats.triangles * 3);
  expect(r.glb.extras.id_instancia).toBe(1);
  expect(r.saved.malla).toBe("artifacts/mallas/synthetic-two-spheres/1.glb");
  expect(r.malla, "instancias.json referencia la malla").toBe("artifacts/mallas/synthetic-two-spheres/1.glb");
  expect(r.status).toMatch(/^synthetic-two-spheres_instancia-1\.glb: .* guardado en artifacts\/mallas/);
  expect(r.restored.isolate, "el aislamiento se restaura").toBe(0);
  expect(r.restored.target, "la cámara vuelve a su objetivo").toEqual([0, 0, 0]);
  expect(calls).toHaveLength(1);
  expect(calls[0].magic).toBe("glTF");
  expect(calls[0].ambito).toBe("instancia");
  expect(calls[0].id_instancia).toBe(1);
  expect(calls[0].metadatos.malla.vertices).toBe(r.stats.vertices);

  await page.selectOption("#mesh-scope", "escena");
  await expect(page.locator("#mesh-build")).toHaveText("Crear GLB de la escena");
  await page.evaluate(() => {
    document.getElementById("mesh-views").value = "8";
    document.getElementById("mesh-resolution").value = "40";
    document.getElementById("mesh-edge").value = "128";
    document.getElementById("mesh-build").click();
  });
  await page.waitForFunction(() => window.__gsMesh.last?.scope === "escena", null, { timeout: 60_000 });
  const scene = await page.evaluate(() => {
    const result = window.__gsMesh.last;
    return {
      name: result.name,
      scope: result.scope,
      label: result.label,
      stats: result.stats,
      metadata: result.metadatos,
      saved: result.saved,
      status: document.getElementById("mesh-status").textContent,
      restored: { isolate: window.__gsRenderer.params.isolateLabel, target: window.__gsCamera.target.map((v) => +v.toFixed(3)) },
    };
  });
  console.log(`[f6-scene] ${JSON.stringify({ ...scene, stats: { ...scene.stats, validationBeforeRepair: undefined, validation: undefined, repair: undefined } })}`);
  expect(scene.name).toBe("synthetic-two-spheres_escena.glb");
  expect(scene.scope).toBe("escena");
  expect(scene.label).toBeNull();
  expect(scene.metadata.ambito).toBe("escena");
  expect(scene.metadata.id_instancia).toBeNull();
  expect(scene.stats.components).toBeGreaterThanOrEqual(2);
  expect(scene.stats.bbox.min[0]).toBeLessThan(-1.4);
  expect(scene.stats.bbox.max[0]).toBeGreaterThan(1.4);
  expect(Math.abs(scene.stats.centroid[0])).toBeLessThan(0.05);
  expect(scene.saved.malla).toBe("artifacts/mallas/synthetic-two-spheres/escena.glb");
  expect(scene.status).toMatch(/^synthetic-two-spheres_escena\.glb: .* guardado en artifacts\/mallas/);
  expect(scene.restored.isolate).toBe(0);
  expect(scene.restored.target).toEqual([0, 0, 0]);
  expect(calls).toHaveLength(2);
  expect(calls[1].ambito).toBe("escena");
  expect(calls[1].id_instancia).toBeNull();
  expect(calls[1].extras.ambito).toBe("escena");
  expect(calls[1].extras.id_instancia).toBeNull();
});

test("Malla: print path repairs topology and writes a millimetre-scale 3MF package", async ({ page }) => {
  const calls = [];
  await page.route(`${SIDECAR}/**`, async (route) => {
    const req = route.request();
    if (req.method() === "OPTIONS") return route.fulfill({ status: 204, headers: CORS });
    const body = req.postDataJSON();
    const file = Buffer.from(body.archivo_b64, "base64");
    calls.push({ ambito: body.ambito, id_instancia: body.id_instancia, formato: body.formato, magic: file.subarray(0, 4).toString("hex"), bytes: file.length, metadata: body.metadatos });
    const stem = body.id_instancia == null ? "escena" : body.id_instancia;
    return route.fulfill({ status: 200, headers: { ...CORS, "Content-Type": "application/json" }, body: JSON.stringify({ ok: true, carpeta: `artifacts/mallas/${body.escena}`, malla: `artifacts/mallas/${body.escena}/${stem}.3mf`, bytes: file.length, metadatos: null }) });
  });
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(() => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsMesh, null, { timeout: 60_000 });
  const r = await page.evaluate(async () => {
    const result = await window.__gsMesh.build(1, { views: 8, resolution: 32, edge: 128, format: "3mf", maxDimensionMm: 80, download: false, save: true, returnMesh: true });
    const { read3mfFiles } = await import("/shared/three-mf.js");
    const files = read3mfFiles(result.threeMf);
    const model = new TextDecoder().decode(files.get("3D/3dmodel.model"));
    const p = result.mesh.positions;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < p.length; i += 3) for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], p[i + a]); max[a] = Math.max(max[a], p[i + a]); }
    return {
      name: result.name,
      format: result.format,
      bytes: result.bytes,
      files: [...files.keys()],
      unitMm: /<model unit="millimeter"/.test(model),
      vertexElements: (model.match(/<vertex /g) || []).length,
      triangleElements: (model.match(/<triangle /g) || []).length,
      min,
      max,
      validation: result.stats.validation,
      repair: result.stats.repair,
      metodo: result.metadatos.metodo,
      impresion: result.metadatos.impresion,
      saved: result.saved,
      status: document.getElementById("mesh-status").textContent,
    };
  });
  console.log(`[f6-3mf] ${JSON.stringify({ ...r, impresion: { ...r.impresion, validacion: undefined } })}`);
  expect(r.name).toBe("synthetic-two-spheres_instancia-1.3mf");
  expect(r.format).toBe("3mf");
  expect(r.files).toEqual(["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]);
  expect(r.unitMm).toBe(true);
  expect(r.vertexElements).toBe(r.validation.vertices);
  expect(r.triangleElements).toBe(r.validation.triangles);
  expect(r.validation.printable).toBe(true);
  expect(r.validation.boundaryEdges).toBe(0);
  expect(r.validation.nonManifoldEdges).toBe(0);
  expect(r.metodo.extraccion).toBe("marching-tetrahedra");
  expect(r.metodo.reparacion).toBe(true);
  expect(r.impresion.unidad).toBe("millimeter");
  expect(r.impresion.dimension_maxima_mm).toBe(80);
  expect(Math.max(...r.max.map((v, a) => v - r.min[a]))).toBeCloseTo(80, 3);
  expect(r.min[2]).toBeCloseTo(0, 5);
  expect(r.saved.malla).toBe("artifacts/mallas/synthetic-two-spheres/1.3mf");
  expect(r.status).toMatch(/^synthetic-two-spheres_instancia-1\.3mf: .* · cerrada ·/);
  expect(calls).toHaveLength(1);
  expect(calls[0].formato).toBe("3mf");
  expect(calls[0].magic).toBe("504b0304");
  expect(calls[0].metadata.impresion.unidad).toBe("millimeter");

  const scene = await page.evaluate(async () => {
    const result = await window.__gsMesh.buildScene({ views: 8, resolution: 40, edge: 128, format: "3mf", maxDimensionMm: 80, download: false, save: true, returnMesh: true });
    const { read3mfFiles } = await import("/shared/three-mf.js");
    const files = read3mfFiles(result.threeMf);
    const model = new TextDecoder().decode(files.get("3D/3dmodel.model"));
    return {
      name: result.name,
      scope: result.scope,
      label: result.label,
      components: result.stats.components,
      printable: result.metadatos.impresion.validacion.printable,
      maxDimension: Math.max(...result.metadatos.impresion.bbox_mm.size),
      saved: result.saved,
      unitMm: /<model unit="millimeter"/.test(model),
    };
  });
  console.log(`[f6-scene-3mf] ${JSON.stringify(scene)}`);
  expect(scene.name).toBe("synthetic-two-spheres_escena.3mf");
  expect(scene.scope).toBe("escena");
  expect(scene.label).toBeNull();
  expect(scene.components).toBeGreaterThanOrEqual(2);
  expect(scene.printable).toBe(true);
  expect(scene.maxDimension).toBeCloseTo(80, 3);
  expect(scene.unitMm).toBe(true);
  expect(scene.saved.malla).toBe("artifacts/mallas/synthetic-two-spheres/escena.3mf");
  expect(calls).toHaveLength(2);
  expect(calls[1].ambito).toBe("escena");
  expect(calls[1].id_instancia).toBeNull();
  expect(calls[1].magic).toBe("504b0304");
});

test("Malla: mesh colour follows the instance colour and the median-depth path also works", async ({ page }) => {
  await page.route(`${SIDECAR}/**`, (route) => route.abort("connectionrefused"));
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(() => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsMesh, null, { timeout: 60_000 });
  await page.waitForFunction(() => window.__gsInstances.project([1, 0, 0]) !== null, null, { timeout: 15_000 });
  const r = await page.evaluate(async () => {
    const r = await window.__gsMesh.build(2, { views: 8, resolution: 40, edge: 128, depth: "mediana", download: false, save: true, returnMesh: true });
    const m = r.mesh;
    let red = 0, blue = 0;
    for (let i = 0; i < m.vertexCount; i++) { red += m.colors[i * 3]; blue += m.colors[i * 3 + 2]; }
    return { vertices: m.vertexCount, red: red / m.vertexCount, blue: blue / m.vertexCount, meanRadius: r.stats.meanRadius, saved: r.saved, status: document.getElementById("mesh-status").textContent, profundidad: r.metadatos.metodo.profundidad };
  });
  console.log(`[f6] esfera B (mediana): ${JSON.stringify(r)}`);
  expect(r.profundidad).toBe("mediana");
  expect(r.vertices).toBeGreaterThan(200);
  expect(r.red, "la esfera B es naranja: rojo alto").toBeGreaterThan(0.6);
  expect(r.blue, "la esfera B es naranja: azul bajo").toBeLessThan(0.4);
  expect(Math.abs(r.meanRadius - 0.5) / 0.5).toBeLessThan(RADIUS_TOLERANCE);
  expect(r.saved).toBeNull();
  expect(r.status).toMatch(/· descarga local/);
});
