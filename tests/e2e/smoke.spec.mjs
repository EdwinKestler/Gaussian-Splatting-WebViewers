// WebGPU smoke tests (plan F0). They run headless on SwiftShader, so they
// never touch a canvas context: every GPU check uses offscreen resources
// and reads results back through GPUBuffer.mapAsync.

import { test, expect } from "@playwright/test";

const DEMO_PLY_URL = "/gaussian_splatting_webgpu/demo.ply";
const SPLAT_IO_URL = "/shared/splat-io.js";
const DEMO_PLY_COUNT = 4292;
const GAUSSIAN_STRIDE = 12;
const SH_STRIDE = 48;

/** Forward browser console output to the test log (prefixed for grepping). */
function pipeConsole(page) {
  page.on("console", (msg) => {
    // Errors carry the URL that produced them (e.g. Chromium's automatic /favicon.ico probe).
    const where = msg.type() === "error" && msg.location().url ? ` (${msg.location().url})` : "";
    console.log(`[browser:${msg.type()}] ${msg.text()}${where}`);
  });
  page.on("pageerror", (err) => console.log(`[browser:pageerror] ${err.message}`));
}

test.beforeEach(async ({ page }) => {
  pipeConsole(page);
  await page.goto("/index.html");
});

test("WebGPU adapter available", async ({ page }, testInfo) => {
  const result = await page.evaluate(async () => {
    if (!navigator.gpu) return { hasGpu: false, adapter: null };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { hasGpu: true, adapter: null };
    const info = adapter.info ?? {};
    return {
      hasGpu: true,
      adapter: {
        vendor: info.vendor ?? "",
        architecture: info.architecture ?? "",
        device: info.device ?? "",
        description: info.description ?? "",
        features: [...adapter.features].sort(),
      },
    };
  });
  expect(result.hasGpu, "navigator.gpu debe existir (¿se sirve por http://?)").toBe(true);
  expect(result.adapter, "requestAdapter() devolvió null").not.toBeNull();
  const summary = `vendor=${result.adapter.vendor} architecture=${result.adapter.architecture}`;
  testInfo.annotations.push({ type: "webgpu-adapter", description: summary });
  console.log(`[smoke] adaptador WebGPU: ${summary}`);
});

test("shared/splat-io parses demo.ply in browser", async ({ page }) => {
  const result = await page.evaluate(
    async ({ plyUrl, ioUrl, gaussianStride, shStride }) => {
      const io = await import(ioUrl);
      const response = await fetch(plyUrl);
      if (!response.ok) throw new Error(`fetch ${plyUrl} -> HTTP ${response.status}`);
      const buffer = await response.arrayBuffer();

      const header = io.describePly(buffer);
      const cloud = io.toGaussianCloud(buffer, "demo.ply");

      // Cheap invariants: opacities are sigmoid outputs, quaternions are unit.
      let badOpacity = 0;
      let badQuat = 0;
      for (let i = 0; i < cloud.count; i++) {
        const g = i * gaussianStride;
        const op = cloud.gaussians[g + 3];
        if (!(op >= 0 && op <= 1)) badOpacity++;
        const qn = Math.hypot(
          cloud.gaussians[g + 8],
          cloud.gaussians[g + 9],
          cloud.gaussians[g + 10],
          cloud.gaussians[g + 11]
        );
        if (Math.abs(qn - 1) > 1e-3) badQuat++;
      }
      return {
        byteLength: buffer.byteLength,
        header,
        count: cloud.count,
        shDegree: cloud.shDegree,
        format: cloud.format,
        variant: cloud.variant,
        gaussiansLength: cloud.gaussians.length,
        shLength: cloud.sh.length,
        expectedGaussians: cloud.count * gaussianStride,
        expectedSh: cloud.count * shStride,
        badOpacity,
        badQuat,
        bounds: cloud.bounds,
      };
    },
    { plyUrl: DEMO_PLY_URL, ioUrl: SPLAT_IO_URL, gaussianStride: GAUSSIAN_STRIDE, shStride: SH_STRIDE }
  );

  console.log(
    `[smoke] demo.ply: ${result.byteLength} bytes, ${result.count} gaussianas, SH${result.shDegree}, ${result.variant}`
  );
  expect(result.header.vertexCount).toBe(DEMO_PLY_COUNT);
  expect(result.header.shDegree).toBe(0);
  expect(result.header.encoding).toBe("binary_le");
  expect(result.count).toBe(DEMO_PLY_COUNT);
  expect(result.shDegree).toBe(0);
  expect(result.format).toBe("ply");
  expect(result.variant).toBe("3dgs");
  expect(result.gaussiansLength).toBe(result.expectedGaussians);
  expect(result.shLength).toBe(result.expectedSh);
  expect(result.badOpacity, "opacidades fuera de [0,1]").toBe(0);
  expect(result.badQuat, "cuaterniones no unitarios").toBe(0);
  expect(result.bounds).toBeTruthy();
});

test("compute + offscreen render survive under SwiftShader", async ({ page }) => {
  const result = await page.evaluate(async () => {
    const WORKGROUPS = 64;
    const WORKGROUP_SIZE = 64;
    const TEX_SIZE = 8;
    const BYTES_PER_ROW = 256; // WebGPU minimum alignment for copyTextureToBuffer

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("requestAdapter() devolvió null");
    const device = await adapter.requestDevice();

    // Device-loss and uncaptured-error bookkeeping.
    let lost = null;
    device.lost.then((info) => { lost = { reason: info.reason, message: info.message }; });
    const uncaptured = [];
    device.addEventListener("uncapturederror", (ev) => uncaptured.push(String(ev.error?.message ?? ev.error)));

    async function readBuffer(src, byteLength) {
      const staging = device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(src, 0, staging, 0, byteLength);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      return copy;
    }

    // (1) Compute pass with atomics on a storage buffer.
    async function runAtomicCompute() {
      const counter = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(counter, 0, new Uint32Array([0]));
      const module = device.createShaderModule({
        code: `
          @group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
          @compute @workgroup_size(${WORKGROUP_SIZE})
          fn main() { atomicAdd(&counter, 1u); }
        `,
      });
      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: counter } }],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(WORKGROUPS);
      pass.end();
      device.queue.submit([encoder.finish()]);
      const data = new Uint32Array(await readBuffer(counter, 4));
      return data[0];
    }

    // (2) Clear + draw into an offscreen rgba8unorm texture, then read it back.
    async function runOffscreenRender() {
      const texture = device.createTexture({
        size: { width: TEX_SIZE, height: TEX_SIZE },
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      const module = device.createShaderModule({
        code: `
          // Triangle covering the bottom-right quadrant (clip space y down = bottom).
          @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
            var p = array<vec2f, 3>(vec2f(0.0, 0.0), vec2f(3.0, 0.0), vec2f(0.0, -3.0));
            return vec4f(p[i], 0.0, 1.0);
          }
          @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
        `,
      });
      const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
        primitive: { topology: "triangle-list" },
      });
      const readback = device.createBuffer({
        size: BYTES_PER_ROW * TEX_SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          clearValue: { r: 1, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.draw(3);
      pass.end();
      encoder.copyTextureToBuffer(
        { texture },
        { buffer: readback, bytesPerRow: BYTES_PER_ROW },
        { width: TEX_SIZE, height: TEX_SIZE }
      );
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const pixels = new Uint8Array(readback.getMappedRange().slice(0));
      readback.unmap();
      const px = (x, y) => Array.from(pixels.subarray(y * BYTES_PER_ROW + x * 4, y * BYTES_PER_ROW + x * 4 + 4));
      return { topLeft: px(0, 0), bottomRight: px(TEX_SIZE - 1, TEX_SIZE - 1) };
    }

    device.pushErrorScope("validation");
    const atomicTotal = await runAtomicCompute();
    const pixels = await runOffscreenRender();
    await device.queue.onSubmittedWorkDone();
    const validationError = await device.popErrorScope();
    // Give a pending device-loss promise a chance to settle before we look.
    await new Promise((resolve) => setTimeout(resolve, 250));

    return {
      expectedAtomicTotal: WORKGROUPS * WORKGROUP_SIZE,
      atomicTotal,
      pixels,
      lost,
      uncaptured,
      validationError: validationError ? validationError.message : null,
    };
  });

  console.log(`[smoke] atomicAdd total=${result.atomicTotal} pixels=${JSON.stringify(result.pixels)}`);
  expect(result.validationError, "error de validación WebGPU").toBeNull();
  expect(result.uncaptured, "errores no capturados de WebGPU").toEqual([]);
  expect(result.atomicTotal).toBe(result.expectedAtomicTotal);
  expect(result.pixels.topLeft).toEqual([255, 0, 0, 255]);
  expect(result.pixels.bottomRight).toEqual([0, 255, 0, 255]);
  expect(result.lost, "el dispositivo WebGPU se perdió").toBeNull();
});
