#!/usr/bin/env node
// One-shot: render a Mermaid diagram to PNG (and SVG) with the local `mermaid`
// package and Playwright's Chromium, without mermaid-cli or network access.
//   npm install --no-save mermaid          # once (not a project dependency)
//   node scripts/render-mermaid.mjs docs/figures/segmentation-pipeline.mmd docs/figures/segmentation-pipeline.png
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { chromium } from "@playwright/test";

const [src, out] = process.argv.slice(2);
if (!src || !out) {
  console.error("uso: node scripts/render-mermaid.mjs <diagrama.mmd> <salida.png> [escala]");
  process.exit(2);
}
const scale = Number(process.argv[4] || 2);
const ROOT = resolve(new URL("..", import.meta.url).pathname);
const MIME = { ".mjs": "text/javascript", ".js": "text/javascript", ".html": "text/html", ".json": "application/json" };

const diagram = await readFile(src, "utf8");
const html = `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">
<div id="d"></div>
<script type="module">
  import mermaid from "/node_modules/mermaid/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: false, theme: "neutral", flowchart: { htmlLabels: true, curve: "basis" }, themeVariables: { fontFamily: "Inter, Helvetica, Arial, sans-serif", fontSize: "14px" } });
  const { svg } = await mermaid.render("g", ${JSON.stringify(diagram)});
  document.getElementById("d").innerHTML = svg;
  window.__svg = svg;
</script></body>`;

const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path === "/") { res.setHeader("Content-Type", "text/html"); res.end(html); return; }
  try {
    const data = await readFile(resolve(ROOT, "." + path));
    res.setHeader("Content-Type", MIME[extname(path)] || "application/octet-stream");
    res.end(data);
  } catch { res.statusCode = 404; res.end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const browser = await chromium.launch({ headless: true, channel: "chromium" });
try {
  const page = await browser.newPage({ deviceScaleFactor: scale, viewport: { width: 1400, height: 1000 } });
  page.on("pageerror", (e) => console.error("[mermaid]", e.message));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => !!window.__svg, null, { timeout: 30_000 });
  const svgEl = page.locator("#d svg");
  await svgEl.screenshot({ path: out, omitBackground: false });
  const svg = await page.evaluate(() => window.__svg);
  await writeFile(out.replace(/\.png$/i, ".svg"), svg, "utf8");
  console.log(`[render-mermaid] ${out} (+ .svg) ×${scale}`);
} finally {
  await browser.close();
  server.close();
}
