/**
 * F3 K-buffer contribution pass (plan §3.2.A "Pase de contribución").
 *
 * For one camera and one 2D mask (u32 label per pixel, 0 = fondo) it computes
 * for every gaussian i and mask label l the accumulated blending weight
 *   C[i][l] = Σ_{pixels p with mask[p] = l}  α_i(p) · T_i(p)
 * where T_i is the front-to-back transmittance in front of i — exactly the
 * quantity FlashSplat's closed-form assignment needs. As a by-product it
 * resolves the 2DGS *median* depth (first sample where T drops below 0.5)
 * and the accumulated alpha per pixel.
 *
 * Exactness with bounded memory: the globally depth-sorted gaussian list is
 * drawn in contiguous chunks (front to back). For each chunk a fragment pass
 * (fs_contrib in gpu-renderer.js) appends (gaussian index + 1, alpha, depth)
 * to a fixed-size per-pixel list of K entries; a compute pass sorts each list
 * by depth, continues the per-pixel transmittance carried over from the
 * previous chunks, atomicAdds fixed-point weights into a dense N × L_batch
 * contribution matrix, and updates the median depth. A chunk in which any
 * pixel received more than K fragments is split in half and redrawn, so no
 * fragment is ever dropped (a chunk of one gaussian cannot overflow).
 * Labels are processed in batches so the matrix stays under a memory cap.
 */

const WG = 256;
export const KMAX = 16;
export const CONTRIB_SCALE = 4096;
/** Cap for the N × L_batch matrix (bytes); labels are batched to respect it. */
const MAX_CONTRIB_BYTES = 64 * 1024 * 1024;
/** First chunking of the sorted list; chunks halve on overflow and grow back after success. */
const INITIAL_CHUNKS = 4;

const RESOLVE_SHADER = /* wgsl */ `
struct RParams {
  width: u32,
  height: u32,
  k: u32,
  label_base: u32,
  label_count: u32,
  scale: f32,
  _p0: u32,
  _p1: u32,
};
struct KEntry {
  index: u32,
  alpha: f32,
  depth: f32,
  pad: u32,
};
@group(0) @binding(0) var<uniform> p: RParams;
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> entries: array<KEntry>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> contrib: array<atomic<u32>>;
// per-pixel median depth (0 = not reached yet) and transmittance, carried across chunks
@group(0) @binding(5) var<storage, read_write> median_depth: array<f32>;
@group(0) @binding(6) var<storage, read_write> trans: array<f32>;
@group(0) @binding(7) var<storage, read_write> overflow: atomic<u32>;

@compute @workgroup_size(${WG})
fn check_overflow(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.x;
  if (pixel >= p.width * p.height) {
    return;
  }
  if (counts[pixel] > p.k) {
    atomicStore(&overflow, 1u);
  }
}

@compute @workgroup_size(${WG})
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
  let pixel = gid.x;
  if (pixel >= p.width * p.height) {
    return;
  }
  let n = min(counts[pixel], p.k);
  var idx: array<u32, ${KMAX}>;
  var al: array<f32, ${KMAX}>;
  var de: array<f32, ${KMAX}>;
  // insertion sort by depth (front first)
  for (var i = 0u; i < n; i = i + 1u) {
    let e = entries[pixel * p.k + i];
    var j = i;
    loop {
      if (j == 0u) { break; }
      if (de[j - 1u] <= e.depth) { break; }
      idx[j] = idx[j - 1u];
      al[j] = al[j - 1u];
      de[j] = de[j - 1u];
      j = j - 1u;
    }
    idx[j] = e.index;
    al[j] = e.alpha;
    de[j] = e.depth;
  }
  let label = mask[pixel];
  let in_batch = label >= p.label_base && label < p.label_base + p.label_count;
  var t = trans[pixel];
  var median = median_depth[pixel];
  for (var i = 0u; i < n; i = i + 1u) {
    let w = al[i] * t;
    if (in_batch && idx[i] > 0u) {
      let col = label - p.label_base;
      atomicAdd(&contrib[(idx[i] - 1u) * p.label_count + col], u32(w * p.scale + 0.5));
    }
    t = t * (1.0 - al[i]);
    if (median == 0.0 && t < 0.5) {
      median = de[i];
    }
  }
  median_depth[pixel] = median;
  trans[pixel] = t;
}
`;

function align256(bytes) {
  return Math.ceil(bytes / 256) * 256;
}

function isPosInt(v) {
  return Number.isInteger(v) && v > 0;
}

export class ContributionPass {
  /** @param {import("./gpu-renderer.js").WebGPUSplatRenderer} renderer initialised renderer */
  constructor(renderer) {
    this.renderer = renderer;
    const device = renderer.device;
    this.device = device;
    this.width = 0;
    this.height = 0;
    this.k = 0;
    this.batchCapacity = 0; // N * L_batch u32 cells of the contrib buffer

    this.kParamsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rParamsBuffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.kBindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "storage" } },
      ],
    });
    this.kPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderer.renderBindLayout, this.kBindLayout] }),
      vertex: renderer._vertexState("vs_main"),
      fragment: {
        module: renderer.renderModule,
        entryPoint: "fs_contrib",
        targets: [{ format: "r8unorm", writeMask: 0 }],
      },
      primitive: { topology: "triangle-strip" },
    });

    const resolveModule = device.createShaderModule({ code: RESOLVE_SHADER });
    this.resolveBindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const resolveLayout = device.createPipelineLayout({ bindGroupLayouts: [this.resolveBindLayout] });
    this.resolvePipeline = device.createComputePipeline({ layout: resolveLayout, compute: { module: resolveModule, entryPoint: "resolve" } });
    this.checkPipeline = device.createComputePipeline({ layout: resolveLayout, compute: { module: resolveModule, entryPoint: "check_overflow" } });
    this.overflowBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.overflowReadBuffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  /** (Re)allocate the per-pixel buffers for a resolution and K. */
  _ensurePixelBuffers(width, height, k) {
    if (this.width === width && this.height === height && this.k === k) return;
    for (const name of ["countsBuffer", "entriesBuffer", "maskBuffer", "depthBuffer", "transBuffer", "pixelReadBuffer", "dummyTexture"]) {
      if (this[name]) {
        this[name].destroy();
        this[name] = null;
      }
    }
    const device = this.device;
    const pixels = width * height;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.countsBuffer = device.createBuffer({ size: align256(pixels * 4), usage: storage });
    this.entriesBuffer = device.createBuffer({ size: align256(pixels * k * 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.maskBuffer = device.createBuffer({ size: align256(pixels * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.depthBuffer = device.createBuffer({ size: align256(pixels * 4), usage: storage });
    this.transBuffer = device.createBuffer({ size: align256(pixels * 4), usage: storage });
    this.ones = new Float32Array(pixels).fill(1);
    // one read buffer holding [median depth | transmittance]
    this.pixelReadBuffer = device.createBuffer({ size: align256(pixels * 4) * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.dummyTexture = device.createTexture({ size: [width, height], format: "r8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.dummyView = this.dummyTexture.createView();
    this.width = width;
    this.height = height;
    this.k = k;
    device.queue.writeBuffer(this.kParamsBuffer, 0, new Uint32Array([width, height, k, 0]));
    this.kBindGroup = device.createBindGroup({
      layout: this.kBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.kParamsBuffer } },
        { binding: 1, resource: { buffer: this.countsBuffer } },
        { binding: 2, resource: { buffer: this.entriesBuffer } },
      ],
    });
    this.resolveBindGroup = null;
  }

  _ensureContribBuffer(cells) {
    if (this.contribBuffer && this.batchCapacity >= cells) return;
    if (this.contribBuffer) {
      this.contribBuffer.destroy();
      this.contribReadBuffer.destroy();
    }
    const bytes = align256(cells * 4);
    this.contribBuffer = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.contribReadBuffer = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.batchCapacity = cells;
    this.resolveBindGroup = null;
  }

  _getResolveBindGroup() {
    if (this.resolveBindGroup) return this.resolveBindGroup;
    this.resolveBindGroup = this.device.createBindGroup({
      layout: this.resolveBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.rParamsBuffer } },
        { binding: 1, resource: { buffer: this.countsBuffer } },
        { binding: 2, resource: { buffer: this.entriesBuffer } },
        { binding: 3, resource: { buffer: this.maskBuffer } },
        { binding: 4, resource: { buffer: this.contribBuffer } },
        { binding: 5, resource: { buffer: this.depthBuffer } },
        { binding: 6, resource: { buffer: this.transBuffer } },
        { binding: 7, resource: { buffer: this.overflowBuffer } },
      ],
    });
    return this.resolveBindGroup;
  }

  /**
   * Run the pass at the renderer's current camera.
   * @param {{mask:Uint32Array, width:number, height:number, labelCount:number, k?:number}} opts
   *   mask: label per pixel (row-major, top-left origin), values < labelCount; 0 = fondo
   *   labelCount: number of columns of the result (label 0 included)
   * @returns {Promise<{count:number, labelCount:number, contrib:Float32Array, medianDepth:Float32Array,
   *   alpha:Float32Array, overflowPixels:number, width:number, height:number, k:number,
   *   batches:number, chunks:number, splits:number, camera:object}>}
   *   overflowPixels is always 0 (chunks split until no pixel exceeds K); splits counts the retries
   *   contrib[i * labelCount + l] = Σ α·T over pixels of label l (unscaled floats)
   */
  async run({ mask, width, height, labelCount, k = 12 }) {
    const r = this.renderer;
    if (!isPosInt(width) || !isPosInt(height)) throw new Error(`invalid contribution size ${width}x${height}`);
    if (!isPosInt(k) || k > KMAX) throw new Error(`k must be an integer in [1, ${KMAX}], got ${k}`);
    if (!isPosInt(labelCount)) throw new Error(`labelCount must be a positive integer, got ${labelCount}`);
    if (!(mask instanceof Uint32Array) || mask.length !== width * height) {
      throw new Error(`mask must be a Uint32Array of ${width * height} labels`);
    }
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] >= labelCount) throw new Error(`mask[${i}] = ${mask[i]} >= labelCount ${labelCount}`);
    }
    const count = r.count;
    const pixels = width * height;
    const result = {
      count,
      labelCount,
      contrib: new Float32Array(count * labelCount),
      medianDepth: new Float32Array(pixels),
      alpha: new Float32Array(pixels),
      overflowPixels: 0,
      width,
      height,
      k,
      batches: 0,
      chunks: 0,
      splits: 0,
      camera: null,
    };
    this._ensurePixelBuffers(width, height, k);
    this.device.queue.writeBuffer(this.maskBuffer, 0, mask);

    // camera exactly like renderOffscreen(): focal rescaled to the target size
    const vp = r.camera.viewport;
    const focal = [r.camera.focal[0] * (width / Math.max(vp[0], 1e-6)), r.camera.focal[1] * (height / Math.max(vp[1], 1e-6))];
    const viewport = [width, height];
    result.camera = { projection: r.camera.projection.slice(), view: r.camera.view.slice(), focal, viewport, eye: Array.from(r.camera.eye) };
    if (count === 0) return result;

    // one global depth sort for this camera; chunks index the sorted list
    r._prepareFrame(0, focal, viewport);
    {
      const encoder = this.device.createCommandEncoder();
      if (r.sortBindGroup) r._encodeSort(encoder);
      this.device.queue.submit([encoder.finish()]);
    }

    const maxBind = Math.min(MAX_CONTRIB_BYTES, this.device.limits.maxStorageBufferBindingSize || MAX_CONTRIB_BYTES);
    const lb = Math.max(1, Math.min(labelCount, Math.floor(maxBind / (count * 4))));
    this._ensureContribBuffer(count * lb);
    const groups = Math.ceil(pixels / WG);
    const initialChunk = Math.max(1, Math.ceil(count / INITIAL_CHUNKS));
    const zero32 = new Uint32Array(1);

    for (let base = 0; base < labelCount; base += lb) {
      const cols = Math.min(lb, labelCount - base);
      const params = new ArrayBuffer(32);
      new Uint32Array(params, 0, 5).set([width, height, k, base, cols]);
      new Float32Array(params, 20, 1)[0] = CONTRIB_SCALE;
      this.device.queue.writeBuffer(this.rParamsBuffer, 0, params);
      // per batch: contrib = 0, transmittance = 1, median = 0
      this.device.queue.writeBuffer(this.transBuffer, 0, this.ones);
      {
        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.contribBuffer, 0, align256(count * cols * 4));
        encoder.clearBuffer(this.depthBuffer);
        this.device.queue.submit([encoder.finish()]);
      }
      let start = 0;
      let size = initialChunk;
      while (start < count) {
        const end = Math.min(count, start + size);
        // ---- fill the K-buffer with chunk [start, end) of the sorted list
        this.device.queue.writeBuffer(this.overflowBuffer, 0, zero32);
        const encoder = this.device.createCommandEncoder();
        encoder.clearBuffer(this.countsBuffer);
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: this.dummyView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "discard" }],
        });
        pass.setPipeline(this.kPipeline);
        pass.setBindGroup(0, r.renderBindGroup);
        pass.setBindGroup(1, this.kBindGroup);
        pass.setVertexBuffer(0, r.quadBuffer);
        pass.draw(4, end - start, 0, start);
        pass.end();
        const check = encoder.beginComputePass();
        check.setPipeline(this.checkPipeline);
        check.setBindGroup(0, this._getResolveBindGroup());
        check.dispatchWorkgroups(groups);
        check.end();
        encoder.copyBufferToBuffer(this.overflowBuffer, 0, this.overflowReadBuffer, 0, 256);
        this.device.queue.submit([encoder.finish()]);
        await this.overflowReadBuffer.mapAsync(GPUMapMode.READ);
        const overflow = new Uint32Array(this.overflowReadBuffer.getMappedRange())[0];
        this.overflowReadBuffer.unmap();
        if (overflow && size > 1) {
          size = Math.ceil(size / 2);
          result.splits++;
          continue; // redraw the same start with a smaller chunk
        }
        // ---- resolve: continue T per pixel, accumulate, update median
        const enc2 = this.device.createCommandEncoder();
        const resolve = enc2.beginComputePass();
        resolve.setPipeline(this.resolvePipeline);
        resolve.setBindGroup(0, this._getResolveBindGroup());
        resolve.dispatchWorkgroups(groups);
        resolve.end();
        this.device.queue.submit([enc2.finish()]);
        result.chunks++;
        start = end;
        if (!overflow && size < initialChunk) size = Math.min(initialChunk, size * 2);
      }
      // ---- read the batch's columns (and, on the last batch, the per-pixel outputs)
      const last = base + cols >= labelCount;
      const enc3 = this.device.createCommandEncoder();
      enc3.copyBufferToBuffer(this.contribBuffer, 0, this.contribReadBuffer, 0, align256(count * cols * 4));
      if (last) {
        const stride = align256(pixels * 4);
        enc3.copyBufferToBuffer(this.depthBuffer, 0, this.pixelReadBuffer, 0, stride);
        enc3.copyBufferToBuffer(this.transBuffer, 0, this.pixelReadBuffer, stride, stride);
      }
      this.device.queue.submit([enc3.finish()]);
      await this.contribReadBuffer.mapAsync(GPUMapMode.READ);
      try {
        const raw = new Uint32Array(this.contribReadBuffer.getMappedRange(0, count * cols * 4));
        const out = result.contrib;
        const inv = 1 / CONTRIB_SCALE;
        for (let i = 0; i < count; i++) {
          const src = i * cols;
          const dst = i * labelCount + base;
          for (let c = 0; c < cols; c++) out[dst + c] = raw[src + c] * inv;
        }
      } finally {
        this.contribReadBuffer.unmap();
      }
      result.batches++;
      if (last) {
        await this.pixelReadBuffer.mapAsync(GPUMapMode.READ);
        try {
          const stride = align256(pixels * 4);
          const mapped = this.pixelReadBuffer.getMappedRange();
          result.medianDepth.set(new Float32Array(mapped, 0, pixels));
          const t = new Float32Array(mapped, stride, pixels);
          for (let i = 0; i < pixels; i++) result.alpha[i] = 1 - t[i];
        } finally {
          this.pixelReadBuffer.unmap();
        }
      }
    }
    return result;
  }

  destroy() {
    for (const name of ["countsBuffer", "entriesBuffer", "maskBuffer", "depthBuffer", "transBuffer", "pixelReadBuffer", "contribBuffer", "contribReadBuffer", "kParamsBuffer", "rParamsBuffer", "overflowBuffer", "overflowReadBuffer", "dummyTexture"]) {
      if (this[name]) this[name].destroy();
      this[name] = null;
    }
  }
}
