/**
 * WebGPU Gaussian splat renderer.
 *
 * Each frame:
 *   1. Compute view-space depths + min/max
 *   2. Quantize 16-bit keys
 *   3. Histogram + GPU exclusive prefix sum
 *   4. Scatter sorted indices (far → near)
 *   5. Instanced ellipse rasterization with premultiplied alpha
 */

const WG = 256;
const HIST_BINS = 65536;
const HIST_BLOCKS = HIST_BINS / WG;

const UNIFORM_FLOATS = 64; // 256 bytes, 16-byte aligned

const SORT_SHADER = /* wgsl */ `
struct SortUniforms {
  view: mat4x4<f32>,
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> su: SortUniforms;
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read_write> depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> minmax: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(7) var<storage, read_write> sorted: array<u32>;

fn f32_to_sortable(f: f32) -> u32 {
  let u = bitcast<u32>(f);
  let mask = select(0x80000000u, 0xffffffffu, (u & 0x80000000u) != 0u);
  return u ^ mask;
}

fn sortable_to_f32(u: u32) -> f32 {
  let mask = select(0x80000000u, 0xffffffffu, (u & 0x80000000u) == 0u);
  return bitcast<f32>(u ^ mask);
}

@compute @workgroup_size(${WG})
fn compute_depth(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.count) { return; }
  let base = i * 8u;
  let p = vec4<f32>(
    bitcast<f32>(splats[base + 0u]),
    bitcast<f32>(splats[base + 1u]),
    bitcast<f32>(splats[base + 2u]),
    1.0
  );
  let cam = su.view * p;
  let dist = -cam.z;
  depths[i] = dist;
  atomicMin(&minmax[0], f32_to_sortable(dist));
  atomicMax(&minmax[1], f32_to_sortable(dist));
}

@compute @workgroup_size(${WG})
fn compute_keys(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.count) { return; }
  let min_d = sortable_to_f32(atomicLoad(&minmax[0]));
  let max_d = sortable_to_f32(atomicLoad(&minmax[1]));
  let span = max(max_d - min_d, 1e-6);
  let t = clamp((depths[i] - min_d) / span, 0.0, 1.0);
  // Far (t=1) → key 0 so ascending sort draws back-to-front.
  keys[i] = u32((1.0 - t) * 65535.0);
}

@compute @workgroup_size(${WG})
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.count) { return; }
  atomicAdd(&hist[keys[i]], 1u);
}

var<workgroup> scan_shared: array<u32, ${WG}>;
var<workgroup> scan_original: array<u32, ${WG}>;

@compute @workgroup_size(${WG})
fn scan_hist_blocks(
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wg: vec3u
) {
  let i = wg.x * ${WG}u + li;
  let val = atomicLoad(&hist[i]);
  scan_original[li] = val;
  scan_shared[li] = val;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= ${WG}u) { break; }
    var add = 0u;
    if (li >= offset) {
      add = scan_shared[li - offset];
    }
    workgroupBarrier();
    scan_shared[li] += add;
    workgroupBarrier();
    offset *= 2u;
  }

  let exclusive = scan_shared[li] - scan_original[li];
  atomicStore(&hist[i], exclusive);
  if (li == ${WG}u - 1u) {
    block_sums[wg.x] = scan_shared[li];
  }
}

@compute @workgroup_size(${WG})
fn scan_block_sums(@builtin(local_invocation_index) li: u32) {
  let val = block_sums[li];
  scan_original[li] = val;
  scan_shared[li] = val;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= ${WG}u) { break; }
    var add = 0u;
    if (li >= offset) {
      add = scan_shared[li - offset];
    }
    workgroupBarrier();
    scan_shared[li] += add;
    workgroupBarrier();
    offset *= 2u;
  }

  block_sums[li] = scan_shared[li] - scan_original[li];
}

@compute @workgroup_size(${WG})
fn add_block_prefix(
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wg: vec3u
) {
  let i = wg.x * ${WG}u + li;
  let prefix = block_sums[wg.x];
  let v = atomicLoad(&hist[i]) + prefix;
  atomicStore(&hist[i], v);
}

@compute @workgroup_size(${WG})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.count) { return; }
  let dest = atomicAdd(&hist[keys[i]], 1u);
  if (dest < su.count) {
    sorted[dest] = i;
  }
}
`;

const RENDER_SHADER = /* wgsl */ `
struct Uniforms {
  projection: mat4x4<f32>,
  view: mat4x4<f32>,
  focal: vec2<f32>,
  viewport: vec2<f32>,
  render_params: vec4<f32>,
  color_basic: vec4<f32>,
  color_levels: vec4<f32>,
  color_mix: vec4<f32>,
  camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(3) var<storage, read> sh_rest: array<f32>;

struct VSIn {
  @location(0) quad_pos: vec2<f32>,
  @builtin(instance_index) instance_id: u32,
};

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) v_color: vec4<f32>,
  @location(1) v_position: vec2<f32>,
  @location(2) v_mode: f32,
};

@vertex
fn vs_main(input: VSIn) -> VSOut {
  let index = sorted_indices[input.instance_id];
  let base = index * 8u;
  let center = vec3<f32>(
    bitcast<f32>(splats[base + 0u]),
    bitcast<f32>(splats[base + 1u]),
    bitcast<f32>(splats[base + 2u]),
  );
  let scale = vec3<f32>(
    bitcast<f32>(splats[base + 3u]),
    bitcast<f32>(splats[base + 4u]),
    bitcast<f32>(splats[base + 5u]),
  );

  let cam = uniforms.view * vec4<f32>(center, 1.0);
  let pos2d = uniforms.projection * cam;
  let clip = 1.2 * pos2d.w;
  if (pos2d.w <= 0.0 || pos2d.z < -pos2d.w ||
      pos2d.x < -clip || pos2d.x > clip ||
      pos2d.y < -clip || pos2d.y > clip) {
    var culled: VSOut;
    culled.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    culled.v_color = vec4<f32>(0.0);
    culled.v_position = vec2<f32>(0.0);
    culled.v_mode = uniforms.render_params.x;
    return culled;
  }

  let packed_color = splats[base + 6u];
  var color = vec4<f32>(
    f32(packed_color & 0xffu),
    f32((packed_color >> 8u) & 0xffu),
    f32((packed_color >> 16u) & 0xffu),
    f32((packed_color >> 24u) & 0xffu)
  ) / 255.0;

  if (uniforms.camera_pos.w > 0.5) {
    let sh_base = index * 9u;
    let dir = normalize(center - uniforms.camera_pos.xyz);
    let sh1 = vec3<f32>(sh_rest[sh_base + 0u], sh_rest[sh_base + 1u], sh_rest[sh_base + 2u]);
    let sh2 = vec3<f32>(sh_rest[sh_base + 3u], sh_rest[sh_base + 4u], sh_rest[sh_base + 5u]);
    let sh3 = vec3<f32>(sh_rest[sh_base + 6u], sh_rest[sh_base + 7u], sh_rest[sh_base + 8u]);
    let shc1 = 0.4886025119029199;
    color = vec4<f32>(
      clamp(color.rgb + shc1 * (-sh1 * dir.y + sh2 * dir.z - sh3 * dir.x), vec3<f32>(0.0), vec3<f32>(1.0)),
      color.a
    );
  }

  let color_mode = uniforms.color_mix.x;
  let luma = dot(color.rgb, vec3<f32>(0.2989, 0.5870, 0.1140));
  if (color_mode > 2.5) {
    color = vec4<f32>(0.0, luma, 0.0, color.a);
  } else if (color_mode > 1.5) {
    let bw = select(0.1, 1.0, luma >= 0.5);
    color = vec4<f32>(vec3<f32>(bw), color.a);
  } else if (color_mode > 0.5) {
    color = vec4<f32>(vec3<f32>(luma), color.a);
  }

  let center_ndc = pos2d.xy / pos2d.w;
  let point_mode = uniforms.render_params.x;
  var major_axis: vec2<f32>;
  var minor_axis: vec2<f32>;

  if (point_mode > 0.5) {
    let point_size = max(uniforms.render_params.y, 0.5);
    major_axis = vec2<f32>(point_size, 0.0);
    minor_axis = vec2<f32>(0.0, point_size);
  } else {
    let packed_rotation = splats[base + 7u];
    let qw = (f32(packed_rotation & 0xffu) - 128.0) / 128.0;
    let qx = (f32((packed_rotation >> 8u) & 0xffu) - 128.0) / 128.0;
    let qy = (f32((packed_rotation >> 16u) & 0xffu) - 128.0) / 128.0;
    let qz = (f32((packed_rotation >> 24u) & 0xffu) - 128.0) / 128.0;

    let qxqx = qx * qx;
    let qyqy = qy * qy;
    let qzqz = qz * qz;
    let qxqy = qx * qy;
    let qxqz = qx * qz;
    let qyqz = qy * qz;
    let qwqx = qw * qx;
    let qwqy = qw * qy;
    let qwqz = qw * qz;

    let m0 = (1.0 - 2.0 * (qyqy + qzqz)) * scale.x;
    let m1 = (2.0 * (qxqy + qwqz)) * scale.x;
    let m2 = (2.0 * (qxqz - qwqy)) * scale.x;
    let m3 = (2.0 * (qxqy - qwqz)) * scale.y;
    let m4 = (1.0 - 2.0 * (qxqx + qzqz)) * scale.y;
    let m5 = (2.0 * (qyqz + qwqx)) * scale.y;
    let m6 = (2.0 * (qxqz + qwqy)) * scale.z;
    let m7 = (2.0 * (qyqz - qwqx)) * scale.z;
    let m8 = (1.0 - 2.0 * (qxqx + qyqy)) * scale.z;

    let vrk = mat3x3<f32>(
      vec3<f32>(m0 * m0 + m3 * m3 + m6 * m6, m0 * m1 + m3 * m4 + m6 * m7, m0 * m2 + m3 * m5 + m6 * m8),
      vec3<f32>(m0 * m1 + m3 * m4 + m6 * m7, m1 * m1 + m4 * m4 + m7 * m7, m1 * m2 + m4 * m5 + m7 * m8),
      vec3<f32>(m0 * m2 + m3 * m5 + m6 * m8, m1 * m2 + m4 * m5 + m7 * m8, m2 * m2 + m5 * m5 + m8 * m8)
    );

    let z = cam.z;
    if (abs(z) < 0.0001) {
      var rejected: VSOut;
      rejected.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
      rejected.v_color = vec4<f32>(0.0);
      rejected.v_position = vec2<f32>(0.0);
      rejected.v_mode = point_mode;
      return rejected;
    }
    let j = mat3x3<f32>(
      vec3<f32>(uniforms.focal.x / z, 0.0, -(uniforms.focal.x * cam.x) / (z * z)),
      vec3<f32>(0.0, -uniforms.focal.y / z, (uniforms.focal.y * cam.y) / (z * z)),
      vec3<f32>(0.0, 0.0, 0.0)
    );

    let view3 = mat3x3<f32>(
      uniforms.view[0].xyz,
      uniforms.view[1].xyz,
      uniforms.view[2].xyz
    );
    let t = transpose(view3) * j;
    var cov2d = transpose(t) * vrk * t;
    let antialias = max(uniforms.color_mix.w, 0.0);
    cov2d[0][0] += antialias;
    cov2d[1][1] += antialias;

    let mid = (cov2d[0][0] + cov2d[1][1]) * 0.5;
    let radius = length(vec2<f32>((cov2d[0][0] - cov2d[1][1]) * 0.5, cov2d[0][1]));
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);
    let axis_vec = vec2<f32>(cov2d[0][1], lambda1 - cov2d[0][0]);
    let axis_len = length(axis_vec);
    let diag = select(vec2<f32>(1.0, 0.0), axis_vec / max(axis_len, 1e-8), axis_len >= 1e-6);
    let splat_scale = uniforms.render_params.w;
    major_axis = min(sqrt(2.0 * lambda1), 1024.0) * diag * splat_scale;
    minor_axis = min(sqrt(2.0 * lambda2), 1024.0) * vec2<f32>(diag.y, -diag.x) * splat_scale;
  }

  var out: VSOut;
  out.v_color = color;
  out.v_position = input.quad_pos;
  out.v_mode = point_mode;
  out.position = vec4<f32>(
    center_ndc
      + input.quad_pos.x * major_axis / uniforms.viewport
      + input.quad_pos.y * minor_axis / uniforms.viewport,
    0.0,
    1.0
  );
  return out;
}

fn grade(rgb_in: vec3<f32>) -> vec3<f32> {
  let brightness = uniforms.color_basic.x;
  let contrast = max(uniforms.color_basic.y, 0.0);
  let gamma = max(uniforms.color_basic.z, 0.001);
  let intensity = max(uniforms.color_levels.z, 0.0);
  let saturate = max(uniforms.color_levels.w, 0.0);
  var rgb = rgb_in + vec3<f32>(brightness);
  rgb = (rgb - vec3<f32>(0.5)) * contrast + vec3<f32>(0.5);
  rgb = pow(max(rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
  rgb *= intensity;
  let luma = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  rgb = mix(vec3<f32>(luma), rgb, saturate);
  return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let rgb = grade(input.v_color.rgb);
  let alpha_mul = clamp(uniforms.color_basic.w, 0.0, 4.0);
  let discard_r2 = uniforms.color_levels.x;
  let a = -dot(input.v_position, input.v_position);
  if (a < -discard_r2) {
    discard;
  }
  if (input.v_mode > 0.5) {
    let alpha = input.v_color.a * alpha_mul;
    return vec4<f32>(rgb * alpha, alpha);
  }
  let b = exp(a) * input.v_color.a * alpha_mul;
  return vec4<f32>(b * rgb, b);
}
`;

function align256(bytes) {
  return Math.ceil(bytes / 256) * 256;
}

function createBuffer(device, size, usage, mappedAtCreation = false) {
  return device.createBuffer({
    size: align256(Math.max(size, 4)),
    usage,
    mappedAtCreation,
  });
}

export class WebGPUSplatRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.format = "bgra8unorm";
    this.count = 0;
    this.capacity = 0;
    this.params = {
      pointMode: 0,
      pointSize: 3,
      opacity: 1,
      splatScale: 1,
      brightness: 0,
      contrast: 1,
      gamma: 1,
      alpha: 1,
      pixelDiscard: 4,
      intensity: 1,
      saturation: 1,
      colorMode: 0,
      antialias: 0.3,
    };
    this.camera = {
      projection: new Float32Array(16),
      view: new Float32Array(16),
      focal: [1, 1],
      viewport: [1, 1],
      eye: [0, 0, 3],
    };
    this.shDegree = 0;
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU is not available in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");
    this.device = await adapter.requestDevice();
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    this.uniformBuffer = createBuffer(
      this.device,
      UNIFORM_FLOATS * 4,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.sortUniformBuffer = createBuffer(
      this.device,
      80,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    this.histBuffer = createBuffer(
      this.device,
      HIST_BINS * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.blockSumBuffer = createBuffer(
      this.device,
      HIST_BLOCKS * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.minmaxBuffer = createBuffer(
      this.device,
      8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.zeroHist = new Uint8Array(HIST_BINS * 4);
    this.minmaxInit = new Uint32Array([0xffffffff, 0]);

    const quad = new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]);
    this.quadBuffer = createBuffer(
      this.device,
      quad.byteLength,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(this.quadBuffer, 0, quad);

    const sortModule = this.device.createShaderModule({ code: SORT_SHADER });
    const renderModule = this.device.createShaderModule({ code: RENDER_SHADER });

    this.sortBindLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.shBuffer = createBuffer(
      this.device,
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.renderBindLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    const makeCompute = (entryPoint) =>
      this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.sortBindLayout] }),
        compute: { module: sortModule, entryPoint },
      });

    this.pipelines = {
      computeDepth: makeCompute("compute_depth"),
      computeKeys: makeCompute("compute_keys"),
      histogram: makeCompute("histogram"),
      scanHist: makeCompute("scan_hist_blocks"),
      scanBlocks: makeCompute("scan_block_sums"),
      addPrefix: makeCompute("add_block_prefix"),
      scatter: makeCompute("scatter"),
    };

    this.renderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.renderBindLayout] }),
      vertex: {
        module: renderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: 8,
            stepMode: "vertex",
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
          },
        ],
      },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: this.format,
            blend: {
              color: {
                srcFactor: "one-minus-dst-alpha",
                dstFactor: "one",
                operation: "add",
              },
              alpha: {
                srcFactor: "one-minus-dst-alpha",
                dstFactor: "one",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-strip" },
    });

    this.device.lost.then((info) => {
      console.error("WebGPU device lost", info);
    });
  }

  setSplats(packedUint8, extra = {}) {
    const count = packedUint8.byteLength / 32;
    this.count = count;
    this.shDegree = extra.shDegree || 0;
    if (count === 0) return;
    if (count > this.capacity) {
      this.capacity = Math.max(count, Math.ceil(this.capacity * 1.5) || count);
      const splatSize = this.capacity * 32;
      const indexSize = this.capacity * 4;
      this.splatBuffer = createBuffer(
        this.device,
        splatSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
      this.depthBuffer = createBuffer(
        this.device,
        indexSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
      this.keyBuffer = createBuffer(
        this.device,
        indexSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
      this.sortedBuffer = createBuffer(
        this.device,
        indexSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
    }
    this.device.queue.writeBuffer(
      this.splatBuffer,
      0,
      packedUint8.buffer,
      packedUint8.byteOffset,
      packedUint8.byteLength
    );
    const identity = new Uint32Array(count);
    for (let i = 0; i < count; i++) identity[i] = i;
    this.device.queue.writeBuffer(this.sortedBuffer, 0, identity);
    const sh1 = extra.sh1;
    if (this.shDegree >= 1 && sh1 && sh1.length >= count * 9) {
      const shBytes = Math.max(count * 9 * 4, 16);
      if (!this.shBuffer || this.shBuffer.size < shBytes) {
        this.shBuffer = createBuffer(
          this.device,
          shBytes,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        );
      }
      this.device.queue.writeBuffer(this.shBuffer, 0, sh1);
    }
    this._rebuildBindGroups();
  }

  _rebuildBindGroups() {
    if (!this.splatBuffer) return;
    this.sortBindGroup = this.device.createBindGroup({
      layout: this.sortBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sortUniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.depthBuffer } },
        { binding: 3, resource: { buffer: this.keyBuffer } },
        { binding: 4, resource: { buffer: this.minmaxBuffer } },
        { binding: 5, resource: { buffer: this.histBuffer } },
        { binding: 6, resource: { buffer: this.blockSumBuffer } },
        { binding: 7, resource: { buffer: this.sortedBuffer } },
      ],
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.sortedBuffer } },
        { binding: 3, resource: { buffer: this.shBuffer } },
      ],
    });
  }

  setCamera(projection, view, focal, viewport, eye) {
    this.camera.projection.set(projection);
    this.camera.view.set(view);
    this.camera.focal = focal;
    this.camera.viewport = viewport;
    if (eye) this.camera.eye = eye;
  }

  setParams(partial) {
    Object.assign(this.params, partial);
  }

  _writeUniforms() {
    const u = new Float32Array(UNIFORM_FLOATS);
    u.set(this.camera.projection, 0);
    u.set(this.camera.view, 16);
    u[32] = this.camera.focal[0];
    u[33] = this.camera.focal[1];
    u[34] = this.camera.viewport[0];
    u[35] = this.camera.viewport[1];
    u[36] = this.params.pointMode;
    u[37] = this.params.pointSize;
    u[38] = this.params.opacity;
    u[39] = this.params.splatScale;
    u[40] = this.params.brightness;
    u[41] = this.params.contrast;
    u[42] = this.params.gamma;
    u[43] = this.params.alpha;
    u[44] = this.params.pixelDiscard;
    u[45] = 0;
    u[46] = this.params.intensity;
    u[47] = this.params.saturation;
    u[48] = this.params.colorMode;
    u[49] = 0;
    u[50] = 0;
    u[51] = this.params.antialias;
    u[52] = this.camera.eye[0];
    u[53] = this.camera.eye[1];
    u[54] = this.camera.eye[2];
    u[55] = this.shDegree >= 1 ? 1 : 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);

    const su = new ArrayBuffer(80);
    const sf = new Float32Array(su);
    const ui = new Uint32Array(su);
    sf.set(this.camera.view, 0);
    ui[16] = this.count;
    this.device.queue.writeBuffer(this.sortUniformBuffer, 0, su);
  }

  render(clearColor = [0, 0, 0, 0]) {
    if (!this.device || this.count === 0 || !this.sortBindGroup) return;
    this._writeUniforms();
    this.device.queue.writeBuffer(this.histBuffer, 0, this.zeroHist);
    this.device.queue.writeBuffer(this.minmaxBuffer, 0, this.minmaxInit);

    const groups = Math.ceil(this.count / WG);
    const encoder = this.device.createCommandEncoder();
    const dispatch = (pipeline, gx) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.sortBindGroup);
      pass.dispatchWorkgroups(gx);
      pass.end();
    };
    dispatch(this.pipelines.computeDepth, groups);
    dispatch(this.pipelines.computeKeys, groups);
    dispatch(this.pipelines.histogram, groups);
    dispatch(this.pipelines.scanHist, HIST_BLOCKS);
    dispatch(this.pipelines.scanBlocks, 1);
    dispatch(this.pipelines.addPrefix, HIST_BLOCKS);
    dispatch(this.pipelines.scatter, groups);

    const view = this.context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: {
            r: clearColor[0],
            g: clearColor[1],
            b: clearColor[2],
            a: clearColor[3],
          },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroup);
    renderPass.setVertexBuffer(0, this.quadBuffer);
    renderPass.draw(4, this.count);
    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
