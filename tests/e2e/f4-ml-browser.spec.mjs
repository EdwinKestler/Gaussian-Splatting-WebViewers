/**
 * In-browser models (plan F3 "SAM 2 en el navegador" and F4 "CLIP opcional"):
 * SAM 2 masks prompted by projected superpoints lift the two synthetic spheres
 * into exactly 2 instances, CLIP embeds each isolated crop and a text query
 * ranks the right sphere first, and the export carries `embedding_clip`.
 *
 * Opt-in: needs the weights in vendor/ml/ (scripts/download-ml-models.sh) and
 * ML_E2E=1, because SAM 2 on WASM costs ~20 s per view under SwiftShader.
 *   ML_E2E=1 npx playwright test tests/e2e/f4-ml-browser.spec.mjs
 */
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";

const VIEWER_PAGE = "/gaussian_splatting_webgpu/index.html?offscreen=1&scene=synthetic";
const MANIFEST = fileURLToPath(new URL("../../vendor/ml/manifest.json", import.meta.url));
const ENABLED = process.env.ML_E2E === "1";

test.skip(!ENABLED, "ML_E2E=1 activa las pruebas con modelos en el navegador");
test.skip(ENABLED && !existsSync(MANIFEST), "faltan los pesos: ejecutar scripts/download-ml-models.sh");
test.setTimeout(10 * 60_000);

test.beforeEach(async ({ page }) => {
  page.on("console", (msg) => {
    if (msg.type() === "error" || /^\[(ml|segmentación)\]/.test(msg.text())) console.log(`[browser:${msg.type()}] ${msg.text().slice(0, 300)}`);
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
});

test("SAM 2 in the browser lifts the two spheres and CLIP semantic search finds them by colour", async ({ page }) => {
  await page.goto(VIEWER_PAGE);
  await page.waitForFunction(() => window.__gsViewer?.name === "synthetic-two-spheres" && !!window.__gsNames && !!window.__gsSegment, null, { timeout: 60_000 });
  await page.waitForFunction(() => window.__gsInstances.project([-1, 0, 0]) !== null, null, { timeout: 15_000 });

  // The HUD offers the source and the semantic checkbox.
  expect(await page.locator('#seg-source option[value="sam2"]').count()).toBe(1);
  await expect(page.locator("#inst-semantic")).toHaveCount(1);

  const lift = await page.evaluate(async () => {
    if (!window.__gsGroups.result) await window.__gsGroups.compute();
    const r = await window.__gsSegment.lift({ source: "sam2", views: 2, samPrompts: 6 });
    return { globalCount: r.globalCount, merges: r.merges, sam: r.views.map((v) => v.sam), rows: window.__gsInstances.list().map((x) => [x.label, x.count]), status: document.getElementById("seg-status").textContent };
  });
  console.log(`[ml-e2e] levantamiento SAM 2: ${JSON.stringify(lift)}`);
  expect(lift.globalCount, "dos esferas → dos instancias").toBe(2);
  expect(lift.merges, "la segunda vista se asocia con la primera").toBe(2);
  for (const v of lift.sam) {
    expect(v.prompts, "una indicación por superpunto visible").toBe(2);
    expect(v.objects.length).toBe(2);
    for (const o of v.objects) expect(o.score).toBeGreaterThan(0.8);
  }
  expect(lift.rows.map((r) => r[1]).sort()).toEqual([2000, 2000]);
  expect(lift.status).toMatch(/^2 instancias · 2 vistas · 2 fusiones/);

  // Which label is the orange sphere? Look at its isolated crop.
  const colours = await page.evaluate(async () => {
    const out = {};
    for (const label of [1, 2]) {
      const png = await window.__gsNames.crop(label);
      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = `data:image/png;base64,${png}`; });
      const c = document.createElement("canvas"); c.width = img.width; c.height = img.height;
      const ctx = c.getContext("2d", { willReadFrequently: true }); ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      let teal = 0, orange = 0;
      for (let p = 0; p < d.length; p += 4) { const r = d[p], b = d[p + 2]; if (r > 240 && d[p + 1] > 240 && b > 240) continue; if (b > r + 30) teal++; else if (r > b + 40) orange++; }
      out[label] = orange > teal ? "orange" : "teal";
    }
    return out;
  });
  const orangeLabel = +Object.keys(colours).find((l) => colours[l] === "orange");
  const tealLabel = orangeLabel === 1 ? 2 : 1;
  expect(colours[tealLabel]).toBe("teal");

  const emb = await page.evaluate(() => window.__gsNames.embed());
  console.log(`[ml-e2e] embeddings: ${JSON.stringify(emb)}`);
  expect(emb.count).toBe(2);
  expect(emb.dimension).toBe(512);
  await expect(page.locator("#inst-name-status")).toHaveText(/^2 embeddings CLIP \(512 d\)/);

  const orange = await page.evaluate(() => window.__gsNames.searchSemantic("an orange ball"));
  const blue = await page.evaluate(() => window.__gsNames.searchSemantic("a blue sphere"));
  console.log(`[ml-e2e] naranja → ${JSON.stringify(orange)} · azul → ${JSON.stringify(blue)}`);
  expect(orange[0].label, "«orange ball» elige la esfera naranja").toBe(orangeLabel);
  expect(blue[0].label, "«blue sphere» elige la esfera turquesa").toBe(tealLabel);
  expect(await page.evaluate(() => window.__gsInstances.current?.label), "la búsqueda semántica selecciona la mejor").toBe(tealLabel);
  await expect(page.locator("#inst-name-status")).toHaveText(/^Semántica «a blue sphere»: #\d -?\d\.\d{3}/);

  // Enter in the search box with the checkbox on goes through CLIP.
  await page.check("#inst-semantic");
  await page.fill("#inst-search", "an orange ball");
  await page.press("#inst-search", "Enter");
  await page.waitForFunction((l) => window.__gsInstances.current?.label === l, orangeLabel, { timeout: 30_000 });

  const exported = await page.evaluate(() => window.__gsSegment.build().json);
  expect(exported.metodo.mascaras).toBe("sam2");
  expect(exported.embeddings).toEqual({ modelo: "Xenova/clip-vit-base-patch32", dimension: 512 });
  for (const it of exported.instancias) {
    expect(it.embedding_clip.length).toBe(512);
    const norm = Math.sqrt(it.embedding_clip.reduce((s, v) => s + v * v, 0));
    expect(Math.abs(norm - 1)).toBeLessThan(0.02);
  }
});
