#!/usr/bin/env node
/**
 * Reproducible WebGPU profile of the shipped alarm-clock scene.
 *
 * Linux/NVIDIA normally needs a real display because Chromium's headless GPU
 * process can reject the Vulkan context. The default is therefore headed;
 * set WEBGPU_PROFILE_HEADLESS=1 only when the platform supports hardware
 * WebGPU headlessly.
 *
 *   npm run profile:webgpu
 *   WEBGPU_PROFILE_HEADLESS=1 npm run profile:webgpu
 */
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.env.WEBGPU_PROFILE_PORT || 8093);
const baseUrl = process.env.WEBGPU_PROFILE_BASE_URL || `http://127.0.0.1:${port}`;
const headed = process.env.WEBGPU_PROFILE_HEADLESS !== "1";
const runs = Math.max(1, Math.min(20, Number(process.env.WEBGPU_PROFILE_RUNS) || 5));
const meshViews = Math.max(4, Math.min(64, Number(process.env.WEBGPU_PROFILE_MESH_VIEWS) || 24));
const meshResolution = Math.max(16, Math.min(256, Number(process.env.WEBGPU_PROFILE_MESH_RESOLUTION) || 96));
const meshEdge = Math.max(64, Math.min(1024, Number(process.env.WEBGPU_PROFILE_MESH_EDGE) || 256));
const profileSam = process.env.WEBGPU_PROFILE_SAM === "1";
const profile3mf = process.env.WEBGPU_PROFILE_3MF !== "0";
const printSizeMm = Math.max(1, Math.min(1000, Number(process.env.WEBGPU_PROFILE_PRINT_MM) || 100));
const defaultArgs = [
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--enable-features=WebGPU,Vulkan,DefaultANGLEVulkan,VulkanFromANGLE",
  "--use-angle=vulkan",
];
const launchArgs = (process.env.WEBGPU_ARGS || defaultArgs.join(" ")).trim().split(/\s+/).filter(Boolean);

const sleep = (ms) => new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
async function waitForServer(url) {
  let last = null;
  for (let i = 0; i < 50; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
      last = new Error(`HTTP ${response.status}`);
    } catch (error) {
      last = error;
    }
    await sleep(100);
  }
  throw new Error(`profile server did not start at ${url}: ${last?.message || "timeout"}`);
}

let server = null;
let browser = null;
try {
  if (!process.env.WEBGPU_PROFILE_BASE_URL) {
    server = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
      cwd: root,
      stdio: ["ignore", "ignore", "inherit"],
    });
    await waitForServer(`${baseUrl}/index.html`);
  }
  browser = await chromium.launch({ channel: "chromium", headless: !headed, args: launchArgs });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.setDefaultTimeout(120_000);
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") {
      console.error(`[browser:${message.type()}] ${message.text()}`);
    }
  });
  const url = `${baseUrl}/gaussian_splatting_webgpu/index.html?offscreen=1&url=../splats/alarm_clock_generated.splat`;
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__gsViewer?.count === 262144 && window.__gsRenderer && window.__gsGroups && window.__gsMesh);

  const profile = await page.evaluate(async ({ runs, meshViews, meshResolution, meshEdge, profileSam, profile3mf, printSizeMm }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("navigator.gpu exists but requestAdapter() returned null");
    const info = adapter.info || {};
    const { OUTPUT_MODE } = await import("/gaussian_splatting_webgpu/gpu-renderer.js");
    const renderer = window.__gsRenderer;
    const canvas = document.querySelector("canvas");
    const width = 512;
    const height = Math.max(64, Math.round(width * canvas.height / Math.max(canvas.width, 1)));
    const measure = async (fn, warmup = true) => {
      if (warmup) await fn();
      const samples = [];
      for (let i = 0; i < runs; i++) {
        const start = performance.now();
        await fn();
        samples.push(performance.now() - start);
      }
      const sorted = [...samples].sort((a, b) => a - b);
      const percentile = (p) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
      return { samples_ms: samples, min_ms: sorted[0], median_ms: percentile(0.5), p95_ms: percentile(0.95) };
    };

    const color = await measure(() => renderer.renderOffscreen({ mode: OUTPUT_MODE.COLOR, width, height, clearColor: [0, 0, 0, 1] }));
    const depth = await measure(() => renderer.renderOffscreen({ mode: OUTPUT_MODE.DEPTH, width, height }));
    const mask = new Uint32Array(width * height);
    const contribution = await measure(
      () => renderer.renderContributions({ mask, width, height, labelCount: 1, k: 24 }),
      false
    );
    const graphStart = performance.now();
    const graph = await window.__gsGroups.compute();
    const graphWallMs = performance.now() - graphStart;

    let segmentation = null;
    if (profileSam) {
      segmentation = await window.__gsSegment.lift({
        source: "sam2",
        views: 4,
        samPrompts: 8,
        backgroundBias: 0.3,
        diffusion: 5,
      });
    }

    const labels = new Uint32Array(window.__gsViewer.count).fill(1);
    renderer.setLabels(labels);
    renderer.setInstance(1, { visible: true });
    const mesh = await window.__gsMesh.build(1, {
      views: meshViews,
      resolution: meshResolution,
      edge: meshEdge,
      depth: "media",
      carve: true,
      download: false,
      save: false,
    });
    let print3mf = null;
    if (profile3mf) {
      const start = performance.now();
      try {
        print3mf = await window.__gsMesh.build(1, {
          views: meshViews,
          resolution: meshResolution,
          edge: meshEdge,
          depth: "media",
          carve: true,
          repair: true,
          format: "3mf",
          maxDimensionMm: printSizeMm,
          download: false,
          save: false,
        });
      } catch (error) {
        print3mf = { error: error.message, failed_ms: performance.now() - start };
      }
    }

    return {
      captured_at: new Date().toISOString(),
      user_agent: navigator.userAgent,
      adapter: {
        vendor: info.vendor || "",
        architecture: info.architecture || "",
        device: info.device || "",
        description: info.description || "",
        features: [...adapter.features].sort(),
      },
      scene: { ...window.__gsViewer },
      target: { width, height, runs },
      render: { color, depth, contribution },
      graph: { wall_ms: graphWallMs, superpoints: graph.superpointCount, stats: graph.stats },
      segmentation,
      mesh: {
        views: meshViews,
        resolution: meshResolution,
        edge: meshEdge,
        total_ms: mesh.ms,
        bytes: mesh.bytes,
        stats: mesh.stats,
        stage_ms: mesh.metadatos.tiempos_ms,
      },
      print_3mf: print3mf && (print3mf.error ? {
        error: print3mf.error,
        failed_ms: print3mf.failed_ms,
        max_dimension_mm: printSizeMm,
      } : {
        views: meshViews,
        resolution: meshResolution,
        edge: meshEdge,
        max_dimension_mm: printSizeMm,
        total_ms: print3mf.ms,
        bytes: print3mf.bytes,
        stats: print3mf.stats,
        stage_ms: print3mf.metadatos.tiempos_ms,
        print: print3mf.metadatos.impresion,
      }),
    };
  }, { runs, meshViews, meshResolution, meshEdge, profileSam, profile3mf, printSizeMm });

  const slug = profile.captured_at.replace(/[:.]/g, "-");
  const output = resolve(root, "artifacts", "profiles", `webgpu-${slug}.json`);
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify({ ...profile, launch: { headed, args: launchArgs } }, null, 2)}\n`);
  console.log(JSON.stringify({ output, ...profile }, null, 2));
} finally {
  if (browser) await browser.close();
  if (server) server.kill("SIGTERM");
}
