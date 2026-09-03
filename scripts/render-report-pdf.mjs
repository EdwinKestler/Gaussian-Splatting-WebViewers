#!/usr/bin/env node
// One-shot: Markdown → PDF with the local `marked` package and Playwright's
// Chromium (no pandoc / LaTeX). Relative images resolve against the file's folder.
//   npm install --no-save marked        # once (not a project dependency)
//   node scripts/render-report-pdf.mjs docs/open-vocab-3dgs-imagine-pipeline-paper.md docs/open-vocab-3dgs-imagine-pipeline-paper.pdf
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { chromium } from "@playwright/test";
import { marked } from "marked";

const [src, out] = process.argv.slice(2);
if (!src || !out) {
  console.error("uso: node scripts/render-report-pdf.mjs <informe.md> <salida.pdf>");
  process.exit(2);
}
const dir = resolve(dirname(src));
const MIME = { ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".css": "text/css" };
const md = await readFile(src, "utf8");
const body = marked.parse(md, { gfm: true });
const html = `<!doctype html><meta charset="utf-8"><title>${src}</title>
<style>
  body { font: 11pt/1.45 Georgia, "Times New Roman", serif; color: #111; margin: 0; }
  h1 { font-size: 20pt; margin: 0 0 6pt; } h2 { font-size: 15pt; margin: 18pt 0 6pt; page-break-after: avoid; } h3 { font-size: 12.5pt; margin: 12pt 0 4pt; }
  p, li { text-align: justify; } code { font: 9.5pt Menlo, Consolas, monospace; background: #f3f3f3; padding: 0 2px; }
  pre { background: #f3f3f3; padding: 6pt 8pt; font-size: 9pt; white-space: pre-wrap; word-break: break-word; }
  table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6pt 0 10pt; page-break-inside: auto; }
  th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; text-align: left; } th { background: #eee; }
  img { max-width: 100%; max-height: 640pt; display: block; margin: 8pt auto; } em { color: #333; }
  a { color: #1a4f8a; text-decoration: none; word-break: break-all; } hr { border: 0; border-top: 1px solid #ccc; margin: 14pt 0; }
</style><body>${body}</body>`;

const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (path === "/") { res.setHeader("Content-Type", "text/html; charset=utf-8"); res.end(html); return; }
  try {
    res.setHeader("Content-Type", MIME[extname(path)] || "application/octet-stream");
    res.end(await readFile(resolve(dir, "." + path)));
  } catch { res.statusCode = 404; res.end(); }
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ headless: true, channel: "chromium" });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "networkidle" });
  await page.pdf({ path: out, format: "A4", printBackground: true, margin: { top: "18mm", bottom: "18mm", left: "16mm", right: "16mm" }, displayHeaderFooter: true, headerTemplate: "<span></span>", footerTemplate: '<div style="font:8pt Georgia,serif;color:#666;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>' });
  console.log(`[render-report-pdf] ${out}`);
} finally {
  await browser.close();
  server.close();
}
