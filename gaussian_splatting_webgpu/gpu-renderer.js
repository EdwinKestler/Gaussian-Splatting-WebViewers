/**
 * WebGPU 3D Gaussian Splatting rasterizer (Kerbl / graphdeco-inria).
 *
 * Each frame:
 *   1. View-space depths + min/max
 *   2. Quantize 16-bit keys (near → 0 so we draw front-to-back)
 *   3. Histogram + GPU exclusive prefix sum
 *   4. Scatter sorted indices
 *   5. Instanced EWA ellipses, paper SH0–3, α = o · exp(-½ r²)
 */

const WG = 256;
const HIST_BINS = 65536;
const HIST_BLOCKS = HIST_BINS / WG;
const UNIFORM_FLOATS = 64;
const GAUSSIAN_FLOATS = 12;
const SH_FLOATS = 48;

const SORT_SHADER = /* wgsl */ `
struct SortUniforms {
  view: mat4x4<f32>,
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

struct Gaussian {
  pos_opacity: vec4<f32>,
  scale: vec4<f32>,
  quat: vec4<f32>,
};

@group(0) @binding(0) var<uniform> su: SortUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
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
  let p = vec4<f32>(gaussians[i].pos_opacity.xyz, 1.0);
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
  // Near (t=0) → key 0 so ascending sort draws front-to-back (INRIA T compositing).
  keys[i] = u32(t * 65535.0);
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

struct Gaussian {
  pos_opacity: vec4<f32>,
  scale: vec4<f32>,
  quat: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(3) var<storage, read> sh_coeffs: array<f32>;

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

const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2_0: f32 = 1.0925484305920792;
const SH_C2_1: f32 = -1.0925484305920792;
const SH_C2_2: f32 = 0.31539156525252005;
const SH_C2_3: f32 = -1.0925484305920792;
const SH_C2_4: f32 = 0.5462742152960396;
const SH_C3_0: f32 = -0.5900435899266435;
const SH_C3_1: f32 = 2.890611442640554;
const SH_C3_2: f32 = -0.4570457994644658;
const SH_C3_3: f32 = 0.3731763325901154;
const SH_C3_4: f32 = -0.4570457994644658;
const SH_C3_5: f32 = 1.445305721320277;
const SH_C3_6: f32 = -0.5900435899266435;

fn sh_band(base: u32, k: u32) -> vec3<f32> {
  let o = base + k * 3u;
  return vec3<f32>(sh_coeffs[o], sh_coeffs[o + 1u], sh_coeffs[o + 2u]);
}

fn eval_sh(index: u32, dir: vec3<f32>, degree: i32) -> vec3<f32> {
  let base = index * 48u;
  var result = SH_C0 * sh_band(base, 0u);
  if (degree > 0) {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    result = result
      - SH_C1 * y * sh_band(base, 1u)
      + SH_C1 * z * sh_band(base, 2u)
      - SH_C1 * x * sh_band(base, 3u);
    if (degree > 1) {
      let xx = x * x;
      let yy = y * y;
      let zz = z * z;
      let xy = x * y;
      let yz = y * z;
      let xz = x * z;
      result = result
        + SH_C2_0 * xy * sh_band(base, 4u)
        + SH_C2_1 * yz * sh_band(base, 5u)
        + SH_C2_2 * (2.0 * zz - xx - yy) * sh_band(base, 6u)
        + SH_C2_3 * xz * sh_band(base, 7u)
        + SH_C2_4 * (xx - yy) * sh_band(base, 8u);
      if (degree > 2) {
        result = result
          + SH_C3_0 * y * (3.0 * xx - yy) * sh_band(base, 9u)
          + SH_C3_1 * xy * z * sh_band(base, 10u)
          + SH_C3_2 * y * (4.0 * zz - xx - yy) * sh_band(base, 11u)
          + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_band(base, 12u)
          + SH_C3_4 * x * (4.0 * zz - xx - yy) * sh_band(base, 13u)
          + SH_C3_5 * z * (xx - yy) * sh_band(base, 14u)
          + SH_C3_6 * x * (xx - 3.0 * yy) * sh_band(base, 15u);
      }
    }
  }
  return max(result + 0.5, vec3<f32>(0.0));
}

fn cov3d_from_quat(q: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
  let w = q.x;
  let x = q.y;
  let y = q.z;
  let z = q.w;
  let r00 = 1.0 - 2.0 * (y * y + z * z);
  let r10 = 2.0 * (x * y + w * z);
  let r20 = 2.0 * (x * z - w * y);
  let r01 = 2.0 * (x * y - w * z);
  let r11 = 1.0 - 2.0 * (x * x + z * z);
  let r21 = 2.0 * (y * z + w * x);
  let r02 = 2.0 * (x * z + w * y);
  let r12 = 2.0 * (y * z - w * x);
  let r22 = 1.0 - 2.0 * (x * x + y * y);
  let m = mat3x3<f32>(
    vec3<f32>(r00, r10, r20) * scale.x,
    vec3<f32>(r01, r11, r21) * scale.y,
    vec3<f32>(r02, r12, r22) * scale.z
  );
  return m * transpose(m);
}

fn empty_vertex(mode: f32) -> VSOut {
  var out: VSOut;
  out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  out.v_color = vec4<f32>(0.0);
  out.v_position = vec2<f32>(0.0);
  out.v_mode = mode;
  return out;
}

@vertex
fn vs_main(input: VSIn) -> VSOut {
  let index = sorted_indices[input.instance_id];
  let g = gaussians[index];
  let center = g.pos_opacity.xyz;
  let opacity = g.pos_opacity.w;
  let splat_scale = max(uniforms.render_params.w, 0.0);
  let scale = g.scale.xyz * splat_scale;
  let quat = g.quat;
  let point_mode = uniforms.render_params.x;

  let cam4 = uniforms.view * vec4<f32>(center, 1.0);
  let cam = cam4.xyz;
  let pos2d = uniforms.projection * cam4;
  let clip = 1.2 * pos2d.w;
  if (pos2d.w <= 0.0 || pos2d.z < -pos2d.w ||
      pos2d.x < -clip || pos2d.x > clip ||
      pos2d.y < -clip || pos2d.y > clip) {
    return empty_vertex(point_mode);
  }

  let degree = i32(uniforms.camera_pos.w);
  let dir = normalize(center - uniforms.camera_pos.xyz);
  var rgb = eval_sh(index, dir, degree);

  let color_mode = uniforms.color_mix.x;
  let luma = dot(rgb, vec3<f32>(0.2989, 0.5870, 0.1140));
  if (color_mode > 2.5) {
    rgb = vec3<f32>(0.0, luma, 0.0);
  } else if (color_mode > 1.5) {
    let bw = select(0.1, 1.0, luma >= 0.5);
    rgb = vec3<f32>(bw);
  } else if (color_mode > 0.5) {
    rgb = vec3<f32>(luma);
  }

  let center_ndc = pos2d.xy / pos2d.w;
  var major_axis: vec2<f32>;
  var minor_axis: vec2<f32>;
  var sigma_quad = input.quad_pos * 3.0;

  if (point_mode > 0.5) {
    let point_size = max(uniforms.render_params.y, 0.5);
    major_axis = vec2<f32>(point_size, 0.0);
    minor_axis = vec2<f32>(0.0, point_size);
    sigma_quad = input.quad_pos;
  } else {
    let z = cam.z;
    if (abs(z) < 0.0001) {
      return empty_vertex(point_mode);
    }
    let vrk = cov3d_from_quat(quat, scale);
    let R = mat3x3<f32>(
      uniforms.view[0].xyz,
      uniforms.view[1].xyz,
      uniforms.view[2].xyz
    );
    let J = mat3x3<f32>(
      vec3<f32>(uniforms.focal.x / z, 0.0, 0.0),
      vec3<f32>(0.0, uniforms.focal.y / z, 0.0),
      vec3<f32>(
        -(uniforms.focal.x * cam.x) / (z * z),
        -(uniforms.focal.y * cam.y) / (z * z),
        0.0
      )
    );
    let view_cov = R * vrk * transpose(R);
    var cov = J * view_cov * transpose(J);
    let antialias = max(uniforms.color_mix.w, 0.0);
    cov[0][0] += antialias;
    cov[1][1] += antialias;

    let a = cov[0][0];
    let b = cov[0][1];
    let c = cov[1][1];
    let mid = 0.5 * (a + c);
    let det = max(a * c - b * b, 0.0);
    let radius = sqrt(max(0.1, mid * mid - det));
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);
    let axis_vec = vec2<f32>(b, lambda1 - a);
    let axis_len = length(axis_vec);
    let diag = select(vec2<f32>(1.0, 0.0), axis_vec / max(axis_len, 1e-8), axis_len >= 1e-6);
    // Paper 3-sigma extent in pixels; quad is [-1, 1].
    major_axis = min(3.0 * sqrt(lambda1), 1024.0) * diag;
    minor_axis = min(3.0 * sqrt(lambda2), 1024.0) * vec2<f32>(diag.y, -diag.x);
  }

  var out: VSOut;
  out.v_color = vec4<f32>(rgb, opacity);
  out.v_position = sigma_quad;
  out.v_mode = point_mode;
  out.position = vec4<f32>(
    center_ndc
      + 2.0 * input.quad_pos.x * major_axis / uniforms.viewport
      + 2.0 * input.quad_pos.y * minor_axis / uniforms.viewport,
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
  let r2 = dot(input.v_position, input.v_position);
  if (r2 > discard_r2) {
    discard;
  }
  if (input.v_mode > 0.5) {
    let alpha = min(0.99, input.v_color.a * alpha_mul);
    return vec4<f32>(rgb * alpha, alpha);
  }
  // Eq. (2): α = o · exp(-½ mahalanobis²), clamped like the CUDA rasterizer.
  let alpha = min(0.99, input.v_color.a * exp(-0.5 * r2) * alpha_mul);
  if (alpha < 1.0 / 255.0) {
    discard;
  }
  return vec4<f32>(rgb * alpha, alpha);
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
      pixelDiscard: 9,
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
    const requiredLimits = {};
    if (adapter.limits.maxStorageBufferBindingSize) {
      requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
    }
    if (adapter.limits.maxBufferSize) {
      requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
    }
    this.device = await adapter.requestDevice({ requiredLimits });
    this.context = this.canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
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

    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
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

  setCloud(gaussians, sh, shDegree = 0) {
    const count = gaussians.length / GAUSSIAN_FLOATS;
    this.count = count;
    this.shDegree = shDegree || 0;
    if (count === 0) return;
    if (count > this.capacity) {
      this.capacity = Math.max(count, Math.ceil(this.capacity * 1.5) || count);
      const gaussSize = this.capacity * GAUSSIAN_FLOATS * 4;
      const shSize = this.capacity * SH_FLOATS * 4;
      const indexSize = this.capacity * 4;
      this.splatBuffer = createBuffer(
        this.device,
        gaussSize,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      );
      this.shBuffer = createBuffer(
        this.device,
        Math.max(shSize, 16),
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
    this.device.queue.writeBuffer(this.splatBuffer, 0, gaussians);
    const shData = sh && sh.length ? sh : new Float32Array(count * SH_FLOATS);
    this.device.queue.writeBuffer(this.shBuffer, 0, shData);
    const identity = new Uint32Array(count);
    for (let i = 0; i < count; i++) identity[i] = i;
    this.device.queue.writeBuffer(this.sortedBuffer, 0, identity);
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
    u[55] = this.shDegree;
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

  async snapshotPng(maxEdge = 1024) {
    if (!this.device || !this.context) {
      throw new Error("WebGPU is not initialized");
    }
    await this.device.queue.onSubmittedWorkDone();
    const canvas = this.canvas;
    if (typeof canvas.toBlob === "function") {
      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("canvas.toBlob failed"))), "image/png");
      });
      if (blob && blob.size > 32) return blob;
    }
    const tex = this.context.getCurrentTexture();
    const w = tex.width;
    const h = tex.height;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    const buf = this.device.createBuffer({
      size: bytesPerRow * h,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, { width: w, height: h });
    this.device.queue.submit([encoder.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const rgba = new Uint8ClampedArray(w * h * 4);
    const bgra = this.format.startsWith("bgra");
    for (let y = 0; y < h; y++) {
      const row = y * bytesPerRow;
      for (let x = 0; x < w; x++) {
        const s = row + x * 4;
        const d = (y * w + x) * 4;
        if (bgra) {
          rgba[d] = src[s + 2];
          rgba[d + 1] = src[s + 1];
          rgba[d + 2] = src[s];
          rgba[d + 3] = src[s + 3];
        } else {
          rgba[d] = src[s];
          rgba[d + 1] = src[s + 1];
          rgba[d + 2] = src[s + 2];
          rgba[d + 3] = src[s + 3];
        }
      }
    }
    buf.unmap();
    buf.destroy();
    const off = document.createElement("canvas");
    let dw = w;
    let dh = h;
    if (Math.max(w, h) > maxEdge) {
      const s = maxEdge / Math.max(w, h);
      dw = Math.max(1, Math.round(w * s));
      dh = Math.max(1, Math.round(h * s));
    }
    off.width = dw;
    off.height = dh;
    const ctx = off.getContext("2d");
    const img = new ImageData(rgba, w, h);
    if (dw === w && dh === h) {
      ctx.putImageData(img, 0, 0);
    } else {
      const full = document.createElement("canvas");
      full.width = w;
      full.height = h;
      full.getContext("2d").putImageData(img, 0, 0);
      ctx.drawImage(full, 0, 0, dw, dh);
    }
    return new Promise((resolve, reject) => {
      off.toBlob((b) => (b ? resolve(b) : reject(new Error("snapshot encode failed"))), "image/png");
    });
  }
}
