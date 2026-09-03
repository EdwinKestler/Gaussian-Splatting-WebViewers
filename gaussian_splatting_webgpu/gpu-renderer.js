/**
 * WebGPU 3D Gaussian Splatting rasterizer (Kerbl / graphdeco-inria).
 *
 * Each frame:
 *   1. View-space depths + min/max (instance transforms applied)
 *   2. Quantize 16-bit keys (near → 0 so we draw front-to-back)
 *   3. Histogram + GPU exclusive prefix sum
 *   4. Scatter sorted indices
 *   5. Instanced EWA ellipses, paper SH0–3, α = o · exp(-½ r²)
 *
 * F1 identity layer (plan §3.2.A):
 *   - `labels[i]` (u32 per gaussian, 0 = fondo) indexes the `instances` table
 *     (MAX_INSTANCES records: rigid/affine xform, tint, visible/selected flags).
 *   - Output modes for offscreen rendering: 0 colour, 1 depth, 2 normal, 3 id
 *     (depth is the alpha-weighted mean, not yet the 2DGS median — see renderOffscreen).
 *   - `renderOffscreen()` / `pick()` / `pickRect()` read results back to the CPU.
 *
 * Data layout (see shared/splat-io.js):
 *   gaussians: Float32Array N*12 = [x,y,z, opacity, sx,sy,sz, pad, qw,qx,qy,qz]
 *   sh:        Float32Array N*48 (16 RGB coefficients, DC first)
 */

const WG = 256;
const HIST_BINS = 65536;
const HIST_BLOCKS = HIST_BINS / WG;
const UNIFORM_FLOATS = 64;
const GAUSSIAN_FLOATS = 12;
const SH_FLOATS = 48;

/** Number of records in the instance table; label values must be < MAX_INSTANCES. */
export const MAX_INSTANCES = 4096;
/** Bytes per instance record: mat4x4<f32> (64) + vec4<f32> tint (16) + vec4<u32> flags (16). */
const INSTANCE_BYTES = 96;
const INSTANCE_WORDS = INSTANCE_BYTES / 4;

/** Output modes accepted by renderOffscreen(). */
export const OUTPUT_MODE = Object.freeze({ COLOR: 0, DEPTH: 1, NORMAL: 2, ID: 3 });

/** Accent colour (#7ee0c8) used for the selection highlight. */
const ACCENT_RGB = [0x7e / 255, 0xe0 / 255, 0xc8 / 255];

const SORT_SHADER = /* wgsl */ `
struct SortUniforms {
  view: mat4x4<f32>,
  count: u32,
  // Offset of the key half inside depth_keys (== buffer capacity).
  key_offset: u32,
  _pad1: u32,
  _pad2: u32,
};

struct Gaussian {
  pos_opacity: vec4<f32>,
  scale: vec4<f32>,
  quat: vec4<f32>,
};

struct Instance {
  xform: mat4x4<f32>,
  tint: vec4<f32>,
  flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> su: SortUniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
// depths (f32 bits) at [0, key_offset), 16-bit keys at [key_offset, 2*key_offset):
// one buffer keeps the compute stage within the default 8 storage buffers.
@group(0) @binding(2) var<storage, read_write> depth_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> minmax: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(6) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(7) var<storage, read> labels: array<u32>;
@group(0) @binding(8) var<storage, read> instances: array<Instance>;

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
  // Sort on the *transformed* centre so moved instances blend in the right order.
  let label = min(labels[i], ${MAX_INSTANCES - 1}u);
  let p = instances[label].xform * vec4<f32>(gaussians[i].pos_opacity.xyz, 1.0);
  let cam = su.view * p;
  let dist = -cam.z;
  depth_keys[i] = bitcast<u32>(dist);
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
  let d = bitcast<f32>(depth_keys[i]);
  let t = clamp((d - min_d) / span, 0.0, 1.0);
  // Near (t=0) → key 0 so ascending sort draws front-to-back (INRIA T compositing).
  depth_keys[su.key_offset + i] = u32(t * 65535.0);
}

@compute @workgroup_size(${WG})
fn histogram(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= su.count) { return; }
  atomicAdd(&hist[depth_keys[su.key_offset + i]], 1u);
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
  let dest = atomicAdd(&hist[depth_keys[su.key_offset + i]], 1u);
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
  // x: output mode (0 colour, 1 depth, 2 normal, 3 id), y: isolate label (0 = none)
  mode_params: vec4<u32>,
  // x: selection highlight strength, y: id alpha threshold
  select_params: vec4<f32>,
};

struct Gaussian {
  pos_opacity: vec4<f32>,
  scale: vec4<f32>,
  quat: vec4<f32>,
};

struct Instance {
  xform: mat4x4<f32>,
  tint: vec4<f32>,
  flags: vec4<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(3) var<storage, read> sh_coeffs: array<f32>;
@group(0) @binding(4) var<storage, read> labels: array<u32>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
// F2: superpoint id + 1 per gaussian (0 = sin grupo), coloured by colour mode 4 (Grupos)
@group(0) @binding(6) var<storage, read> groups: array<u32>;

struct VSIn {
  @location(0) quad_pos: vec2<f32>,
  @builtin(instance_index) instance_id: u32,
};

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) v_color: vec4<f32>,
  @location(1) v_position: vec2<f32>,
  @location(2) v_mode: f32,
  // xyz: view-space normal (smallest axis, facing the camera), w: view distance (-cam.z)
  @location(3) v_aux: vec4<f32>,
  // gaussian index + 1 (0 = culled)
  @location(4) @interpolate(flat) v_index: u32,
  // bit 0: instance selected
  @location(5) @interpolate(flat) v_flags: u32,
};

const ACCENT: vec3<f32> = vec3<f32>(${ACCENT_RGB[0]}, ${ACCENT_RGB[1]}, ${ACCENT_RGB[2]});

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

// Rotation matrix (columns) from quaternion stored as (w, x, y, z).
fn quat_to_mat(q: vec4<f32>) -> mat3x3<f32> {
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
  return mat3x3<f32>(
    vec3<f32>(r00, r10, r20),
    vec3<f32>(r01, r11, r21),
    vec3<f32>(r02, r12, r22)
  );
}

// M = R · S; Σ = M · Mᵀ. With an instance transform A the covariance becomes
// (A·M)·(A·M)ᵀ, which is exact for any affine A (rotation, uniform or
// non-uniform scale) — equivalent to composing quaternions and scaling axes
// in the rigid case.
fn scaled_rotation(rot: mat3x3<f32>, scale: vec3<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(rot[0] * scale.x, rot[1] * scale.y, rot[2] * scale.z);
}

fn empty_vertex(mode: f32) -> VSOut {
  var out: VSOut;
  out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
  out.v_color = vec4<f32>(0.0);
  out.v_position = vec2<f32>(0.0);
  out.v_mode = mode;
  out.v_aux = vec4<f32>(0.0);
  out.v_index = 0u;
  out.v_flags = 0u;
  return out;
}

// Shared vertex logic for every output mode (colour/depth/normal/id).
// Golden-ratio hue palette for group ids; mirrors groupColor() in shared/graph.js.
fn group_color(g: u32) -> vec3<f32> {
  if (g == 0u) { return vec3<f32>(0.35); }
  let h = fract(f32(g) * 0.618033988749895);
  let s = 0.65;
  let v = 0.95;
  let hh = h * 6.0;
  let sector = i32(floor(hh)) % 6;
  let f = hh - floor(hh);
  let p = v * (1.0 - s);
  let q = v * (1.0 - f * s);
  let t = v * (1.0 - (1.0 - f) * s);
  switch (sector) {
    case 0: { return vec3<f32>(v, t, p); }
    case 1: { return vec3<f32>(q, v, p); }
    case 2: { return vec3<f32>(p, v, t); }
    case 3: { return vec3<f32>(p, q, v); }
    case 4: { return vec3<f32>(t, p, v); }
    default: { return vec3<f32>(v, p, q); }
  }
}

fn splat_vertex(input: VSIn) -> VSOut {
  let index = sorted_indices[input.instance_id];
  let g = gaussians[index];
  let point_mode = uniforms.render_params.x;

  let label = min(labels[index], ${MAX_INSTANCES - 1}u);
  let inst = instances[label];
  let isolate = uniforms.mode_params.y;
  if (inst.flags.x == 0u || (isolate != 0u && label != isolate)) {
    return empty_vertex(point_mode);
  }

  let A = mat3x3<f32>(inst.xform[0].xyz, inst.xform[1].xyz, inst.xform[2].xyz);
  let center = (inst.xform * vec4<f32>(g.pos_opacity.xyz, 1.0)).xyz;
  let opacity = g.pos_opacity.w;
  let splat_scale = max(uniforms.render_params.w, 0.0);
  let scale = g.scale.xyz * splat_scale;
  let quat = g.quat;

  let cam4 = uniforms.view * vec4<f32>(center, 1.0);
  let cam = cam4.xyz;
  let pos2d = uniforms.projection * cam4;
  let clip = 1.2 * pos2d.w;
  if (pos2d.w <= 0.0 || pos2d.z < -pos2d.w ||
      pos2d.x < -clip || pos2d.x > clip ||
      pos2d.y < -clip || pos2d.y > clip) {
    return empty_vertex(point_mode);
  }

  // Cofactor matrix of A (columns a1×a2, a2×a0, a0×a1) equals det(A)·A⁻ᵀ, so
  // it transforms normals under any affine A (non-uniform scale included)
  // and gives A⁻¹ = cofᵀ / det without a WGSL inverse().
  let cof = mat3x3<f32>(cross(A[1], A[2]), cross(A[2], A[0]), cross(A[0], A[1]));
  let det_a = dot(A[0], cof[0]);

  let degree = i32(uniforms.camera_pos.w);
  // SH coefficients live in the gaussian's own frame: pull the view direction
  // back through A⁻¹ (only the sign of det matters after normalisation).
  let dir_world = center - uniforms.camera_pos.xyz;
  var dir_obj = transpose(cof) * dir_world;
  if (det_a < 0.0) { dir_obj = -dir_obj; }
  let use_obj = abs(det_a) > 1e-12 && dot(dir_obj, dir_obj) > 0.0;
  let dir = normalize(select(dir_world, dir_obj, use_obj));
  var rgb = eval_sh(index, dir, degree);

  let color_mode = uniforms.color_mix.x;
  let luma = dot(rgb, vec3<f32>(0.2989, 0.5870, 0.1140));
  if (color_mode > 3.5) {
    // Grupos: palette by superpoint, shaded a little by the original luma
    rgb = group_color(groups[index]) * (0.6 + 0.4 * luma);
  } else if (color_mode > 2.5) {
    rgb = vec3<f32>(0.0, luma, 0.0);
  } else if (color_mode > 1.5) {
    let bw = select(0.1, 1.0, luma >= 0.5);
    rgb = vec3<f32>(bw);
  } else if (color_mode > 0.5) {
    rgb = vec3<f32>(luma);
  }
  rgb = mix(rgb, inst.tint.rgb, clamp(inst.tint.w, 0.0, 1.0));

  let R = mat3x3<f32>(
    uniforms.view[0].xyz,
    uniforms.view[1].xyz,
    uniforms.view[2].xyz
  );
  let rot = quat_to_mat(quat);

  // Normal: axis of smallest scale (untransformed scale picks the axis), taken
  // to world space through the cofactor (inverse-transpose) of A so non-uniform
  // instance scales keep it perpendicular to the surface, then to view space;
  // flipped to face the camera (so the sign of det(A) is irrelevant here).
  var axis = rot[0];
  var s_min = g.scale.x;
  if (g.scale.y < s_min) { axis = rot[1]; s_min = g.scale.y; }
  if (g.scale.z < s_min) { axis = rot[2]; }
  let n_view = R * (cof * axis);
  let n_len = length(n_view);
  var normal = select(vec3<f32>(0.0, 0.0, 1.0), n_view / n_len, n_len > 1e-8);
  if (dot(normal, cam) > 0.0) {
    normal = -normal;
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
    let m = A * scaled_rotation(rot, scale);
    let vrk = m * transpose(m);
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
  out.v_aux = vec4<f32>(normal, -cam.z);
  out.v_index = index + 1u;
  out.v_flags = select(0u, 1u, inst.flags.y != 0u);
  out.position = vec4<f32>(
    center_ndc
      + 2.0 * input.quad_pos.x * major_axis / uniforms.viewport
      + 2.0 * input.quad_pos.y * minor_axis / uniforms.viewport,
    0.0,
    1.0
  );
  return out;
}

// Blended pass (colour / depth / normal): no depth attachment, z = 0 as before.
@vertex
fn vs_main(input: VSIn) -> VSOut {
  return splat_vertex(input);
}

// ID pass: depth-tested, z = d / (d + 1) (monotonic in view distance, in [0, 1)).
@vertex
fn vs_id(input: VSIn) -> VSOut {
  var out = splat_vertex(input);
  if (out.v_index != 0u) {
    let d = max(out.v_aux.w, 0.0);
    out.position.z = d / (d + 1.0);
  }
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

// Per-fragment alpha; negative means "discard".
fn splat_alpha(input: VSOut) -> f32 {
  let alpha_mul = clamp(uniforms.color_basic.w, 0.0, 4.0);
  let discard_r2 = uniforms.color_levels.x;
  let r2 = dot(input.v_position, input.v_position);
  if (r2 > discard_r2) {
    return -1.0;
  }
  if (input.v_mode > 0.5) {
    return min(0.99, input.v_color.a * alpha_mul);
  }
  // Eq. (2): α = o · exp(-½ mahalanobis²), clamped like the CUDA rasterizer.
  let alpha = min(0.99, input.v_color.a * exp(-0.5 * r2) * alpha_mul);
  if (alpha < 1.0 / 255.0) {
    return -1.0;
  }
  return alpha;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let alpha = splat_alpha(input);
  if (alpha < 0.0) {
    discard;
  }
  let mode = uniforms.mode_params.x;
  if (mode == 1u) {
    // Premultiplied expected depth: readback divides by accumulated alpha.
    return vec4<f32>(input.v_aux.w * alpha, 0.0, 0.0, alpha);
  }
  if (mode == 2u) {
    // Premultiplied normal mapped to [0, 1].
    return vec4<f32>((input.v_aux.xyz * 0.5 + 0.5) * alpha, alpha);
  }
  var rgb = grade(input.v_color.rgb);
  if ((input.v_flags & 1u) != 0u) {
    rgb = mix(rgb, ACCENT, clamp(uniforms.select_params.x, 0.0, 1.0));
  }
  return vec4<f32>(rgb * alpha, alpha);
}

// Opaque ID pass: first gaussian with α ≥ threshold wins (depth test 'less').
@fragment
fn fs_id(input: VSOut) -> @location(0) u32 {
  let alpha = splat_alpha(input);
  if (alpha < 0.0 || alpha < uniforms.select_params.y) {
    discard;
  }
  return input.v_index;
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

function halfToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

function isNonNegInt(v) {
  return Number.isInteger(v) && v >= 0;
}

function assertLabel(label) {
  if (!isNonNegInt(label) || label >= MAX_INSTANCES) {
    throw new Error(`label must be an integer in [0, ${MAX_INSTANCES}), got ${label}`);
  }
}

function assertVec(name, v, len) {
  if (!v || typeof v.length !== "number" || v.length !== len) {
    throw new Error(`${name} must have length ${len}`);
  }
  for (let i = 0; i < len; i++) {
    if (!Number.isFinite(v[i])) throw new Error(`${name}[${i}] is not finite`);
  }
}

const IDENTITY_MAT4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

const TARGET_FORMATS = {
  [OUTPUT_MODE.COLOR]: { format: "rgba8unorm", bpp: 4 },
  [OUTPUT_MODE.ID]: { format: "r32uint", bpp: 4 },
};

/**
 * WebGPU gaussian splat renderer.
 *
 * Public API (stable contract for UI code):
 *   init({ offscreenOnly }) · setCloud · setCamera · setParams · render · snapshotPng
 *   setLabels · getLabels · setLabel · setInstance · getInstance · resetInstances
 *   renderOffscreen · pick · pickRect
 *
 * Extra params accepted by setParams():
 *   isolateLabel (u32, 0 = none), selectionHighlight (0..1, default 0.45),
 *   idAlphaThreshold (0..1, default 0.5).
 */
export class WebGPUSplatRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.device = null;
    this.context = null;
    this.format = "bgra8unorm";
    /** Texture format used for depth/normal offscreen targets. */
    this.floatFormat = "rgba16float";
    this.hasFloat32Blendable = false;
    this.offscreenOnly = false;
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
      isolateLabel: 0,
      selectionHighlight: 0.45,
      idAlphaThreshold: 0.5,
    };
    this.camera = {
      projection: new Float32Array(16),
      view: new Float32Array(16),
      focal: [1, 1],
      viewport: [1, 1],
      eye: [0, 0, 3],
    };
    this.shDegree = 0;

    this._labels = new Uint32Array(0);
    this._groups = new Uint32Array(0);
    this._instanceData = new ArrayBuffer(MAX_INSTANCES * INSTANCE_BYTES);
    this._instanceF32 = new Float32Array(this._instanceData);
    this._instanceU32 = new Uint32Array(this._instanceData);
    for (let i = 0; i < MAX_INSTANCES; i++) this._resetInstanceRecord(i);

    this._uniformData = new ArrayBuffer(UNIFORM_FLOATS * 4);
    this._uniformF32 = new Float32Array(this._uniformData);
    this._uniformU32 = new Uint32Array(this._uniformData);
    this._sortUniformData = new ArrayBuffer(80);

    this._blendPipelines = new Map();
    this._targets = new Map();
    this._queue = Promise.resolve();
    this._stateVersion = 0;
    this._idCache = null;
    this._warnedNoContext = false;
  }

  /**
   * Create the device, pipelines and static buffers.
   * @param {{offscreenOnly?: boolean}} [options] offscreenOnly skips canvas
   *   context configuration (render() becomes a warning no-op; use
   *   renderOffscreen()/pick()). Needed for headless SwiftShader tests.
   */
  async init(options = {}) {
    const offscreenOnly = !!(options && options.offscreenOnly);
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
    const requiredFeatures = [];
    if (adapter.features && adapter.features.has("float32-blendable")) {
      requiredFeatures.push("float32-blendable");
    }
    this.device = await adapter.requestDevice({ requiredLimits, requiredFeatures });
    this.hasFloat32Blendable = this.device.features.has("float32-blendable");
    this.floatFormat = this.hasFloat32Blendable ? "rgba32float" : "rgba16float";
    this.offscreenOnly = offscreenOnly;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    if (offscreenOnly) {
      this.context = null;
    } else {
      this.context = this.canvas.getContext("webgpu");
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: "premultiplied",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
    }

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

    this.labelBuffer = createBuffer(
      this.device,
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.instanceBuffer = createBuffer(
      this.device,
      MAX_INSTANCES * INSTANCE_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    this.device.queue.writeBuffer(this.instanceBuffer, 0, this._instanceData);

    const sortModule = this.device.createShaderModule({ code: SORT_SHADER });
    this.renderModule = this.device.createShaderModule({ code: RENDER_SHADER });

    const computeStorage = (binding, type) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    this.sortBindLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        computeStorage(1, "read-only-storage"),
        computeStorage(2, "storage"),
        computeStorage(3, "storage"),
        computeStorage(4, "storage"),
        computeStorage(5, "storage"),
        computeStorage(6, "storage"),
        computeStorage(7, "read-only-storage"),
        computeStorage(8, "read-only-storage"),
      ],
    });
    this.shBuffer = createBuffer(
      this.device,
      16,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    );
    const vertexStorage = (binding) => ({
      binding,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: "read-only-storage" },
    });
    this.renderBindLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        vertexStorage(1),
        vertexStorage(2),
        vertexStorage(3),
        vertexStorage(4),
        vertexStorage(5),
        vertexStorage(6),
      ],
    });
    this.renderPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.renderBindLayout],
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

    this.idPipeline = this._createIdPipeline();
    this.renderPipeline = offscreenOnly ? null : this._getBlendPipeline(this.format);

    this.device.lost.then((info) => {
      console.error("WebGPU device lost", info);
    });
  }

  _vertexState(entryPoint) {
    return {
      module: this.renderModule,
      entryPoint,
      buffers: [
        {
          arrayStride: 8,
          stepMode: "vertex",
          attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
        },
      ],
    };
  }

  /** Front-to-back "under" blending pipeline for a given colour target format. */
  _getBlendPipeline(format) {
    let pipeline = this._blendPipelines.get(format);
    if (pipeline) return pipeline;
    const blendComponent = { srcFactor: "one-minus-dst-alpha", dstFactor: "one", operation: "add" };
    pipeline = this.device.createRenderPipeline({
      layout: this.renderPipelineLayout,
      vertex: this._vertexState("vs_main"),
      fragment: {
        module: this.renderModule,
        entryPoint: "fs_main",
        targets: [{ format, blend: { color: blendComponent, alpha: blendComponent } }],
      },
      primitive: { topology: "triangle-strip" },
    });
    this._blendPipelines.set(format, pipeline);
    return pipeline;
  }

  _createIdPipeline() {
    return this.device.createRenderPipeline({
      layout: this.renderPipelineLayout,
      vertex: this._vertexState("vs_id"),
      fragment: {
        module: this.renderModule,
        entryPoint: "fs_id",
        targets: [{ format: "r32uint" }],
      },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      primitive: { topology: "triangle-strip" },
    });
  }

  /**
   * Upload a cloud. Resets all labels to 0 (fondo); the instance table is kept.
   * @param {Float32Array} gaussians N*12 floats
   * @param {Float32Array|null} sh N*48 floats (or empty → zeros)
   * @param {number} [shDegree=0]
   */
  setCloud(gaussians, sh, shDegree = 0) {
    if (!this.device) throw new Error("WebGPU is not initialized");
    if (!gaussians || gaussians.length % GAUSSIAN_FLOATS !== 0) {
      throw new Error(`gaussians length must be a multiple of ${GAUSSIAN_FLOATS}`);
    }
    if (sh && sh.length && sh.length < (gaussians.length / GAUSSIAN_FLOATS) * SH_FLOATS) {
      throw new Error(`sh must have ${SH_FLOATS} floats per gaussian`);
    }
    const count = gaussians.length / GAUSSIAN_FLOATS;
    this.count = count;
    this.shDegree = shDegree || 0;
    this._labels = new Uint32Array(count);
    this._groups = new Uint32Array(count);
    this._idCache = null;
    this._stateVersion++;
    if (count === 0) return;
    if (count > this.capacity) {
      this.capacity = Math.max(count, Math.ceil(this.capacity * 1.5) || count);
      const gaussSize = this.capacity * GAUSSIAN_FLOATS * 4;
      const shSize = this.capacity * SH_FLOATS * 4;
      const indexSize = this.capacity * 4;
      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      this.splatBuffer = createBuffer(this.device, gaussSize, storage);
      this.shBuffer = createBuffer(this.device, Math.max(shSize, 16), storage);
      // depths + keys share one buffer (see SORT_SHADER depth_keys).
      this.depthKeyBuffer = createBuffer(this.device, indexSize * 2, storage);
      this.sortedBuffer = createBuffer(this.device, indexSize, storage);
      this.labelBuffer = createBuffer(this.device, indexSize, storage);
      this.groupBuffer = createBuffer(this.device, indexSize, storage);
    }
    this.device.queue.writeBuffer(this.splatBuffer, 0, gaussians);
    const shData = sh && sh.length ? sh : new Float32Array(count * SH_FLOATS);
    this.device.queue.writeBuffer(this.shBuffer, 0, shData);
    const identity = new Uint32Array(count);
    for (let i = 0; i < count; i++) identity[i] = i;
    this.device.queue.writeBuffer(this.sortedBuffer, 0, identity);
    this.device.queue.writeBuffer(this.labelBuffer, 0, this._labels);
    this.device.queue.writeBuffer(this.groupBuffer, 0, this._groups);
    this._rebuildBindGroups();
  }

  _rebuildBindGroups() {
    if (!this.splatBuffer) return;
    this.sortBindGroup = this.device.createBindGroup({
      layout: this.sortBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sortUniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.depthKeyBuffer } },
        { binding: 3, resource: { buffer: this.minmaxBuffer } },
        { binding: 4, resource: { buffer: this.histBuffer } },
        { binding: 5, resource: { buffer: this.blockSumBuffer } },
        { binding: 6, resource: { buffer: this.sortedBuffer } },
        { binding: 7, resource: { buffer: this.labelBuffer } },
        { binding: 8, resource: { buffer: this.instanceBuffer } },
      ],
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.splatBuffer } },
        { binding: 2, resource: { buffer: this.sortedBuffer } },
        { binding: 3, resource: { buffer: this.shBuffer } },
        { binding: 4, resource: { buffer: this.labelBuffer } },
        { binding: 5, resource: { buffer: this.instanceBuffer } },
        { binding: 6, resource: { buffer: this.groupBuffer } },
      ],
    });
  }

  // ---------------------------------------------------------------- labels

  /**
   * Replace every label. `null` resets all gaussians to 0 (fondo).
   * @param {Uint32Array|null} labels length must equal this.count; values < MAX_INSTANCES
   */
  setLabels(labels) {
    if (labels == null) {
      this._labels.fill(0);
    } else {
      if (labels.length !== this.count) {
        throw new Error(`labels length ${labels.length} != count ${this.count}`);
      }
      for (let i = 0; i < labels.length; i++) {
        if (!isNonNegInt(labels[i]) || labels[i] >= MAX_INSTANCES) {
          throw new Error(`labels[${i}] = ${labels[i]} must be an integer in [0, ${MAX_INSTANCES})`);
        }
      }
      this._labels.set(labels);
    }
    this._uploadLabels(0, this.count);
  }

  /** @returns {Uint32Array} copy of the CPU label mirror (length = count). */
  getLabels() {
    return this._labels.slice();
  }

  // ---------------------------------------------------------------- groups (F2)

  /**
   * Replace the per-gaussian group ids shown by colour mode 4 (Grupos):
   * superpoint id + 1, 0 = sin grupo. Any u32 is accepted (groups do not
   * index the instance table). `null` clears.
   * @param {Uint32Array|null} groups length must equal this.count
   */
  setGroups(groups) {
    if (groups == null) {
      this._groups.fill(0);
    } else {
      if (groups.length !== this.count) {
        throw new Error(`groups length ${groups.length} != count ${this.count}`);
      }
      for (let i = 0; i < groups.length; i++) {
        if (!isNonNegInt(groups[i]) || groups[i] > 0xffffffff) {
          throw new Error(`groups[${i}] = ${groups[i]} must be a u32`);
        }
      }
      this._groups.set(groups);
    }
    this._stateVersion++;
    if (this.device && this.groupBuffer && this.count > 0) {
      this.device.queue.writeBuffer(this.groupBuffer, 0, this._groups, 0, this.count);
    }
  }

  /** @returns {Uint32Array} copy of the CPU group mirror (length = count). */
  getGroups() {
    return this._groups.slice();
  }

  /** Group id of one gaussian (0 = sin grupo / out of range). */
  groupOf(index) {
    return isNonNegInt(index) && index < this._groups.length ? this._groups[index] : 0;
  }

  /**
   * Assign `label` to the given gaussian indices; uploads only [min, max] of the range.
   * @param {Uint32Array|number[]} indices
   * @param {number} label
   */
  setLabel(indices, label) {
    assertLabel(label);
    if (!indices || !indices.length) return;
    let lo = this.count;
    let hi = -1;
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      if (!isNonNegInt(i) || i >= this.count) {
        throw new Error(`gaussian index ${i} out of range [0, ${this.count})`);
      }
      this._labels[i] = label;
      if (i < lo) lo = i;
      if (i > hi) hi = i;
    }
    this._uploadLabels(lo, hi + 1);
  }

  _uploadLabels(start, end) {
    this._idCache = null;
    this._stateVersion++;
    if (!this.device || !this.labelBuffer || end <= start) return;
    this.device.queue.writeBuffer(this.labelBuffer, start * 4, this._labels, start, end - start);
  }

  // ------------------------------------------------------------- instances

  _resetInstanceRecord(label) {
    const base = label * INSTANCE_WORDS;
    this._instanceF32.set(IDENTITY_MAT4, base);
    this._instanceF32.fill(0, base + 16, base + 20);
    this._instanceU32[base + 20] = 1; // visible
    this._instanceU32[base + 21] = 0; // selected
    this._instanceU32[base + 22] = 0;
    this._instanceU32[base + 23] = 0;
  }

  _uploadInstance(label) {
    this._idCache = null;
    this._stateVersion++;
    if (!this.device || !this.instanceBuffer) return;
    this.device.queue.writeBuffer(
      this.instanceBuffer,
      label * INSTANCE_BYTES,
      this._instanceData,
      label * INSTANCE_BYTES,
      INSTANCE_BYTES
    );
  }

  /**
   * Merge fields into an instance record and upload it.
   * @param {number} label 0..MAX_INSTANCES-1 (0 = fondo)
   * @param {{xform?: Float32Array|number[], tint?: number[], visible?: boolean, selected?: boolean}} fields
   *   xform: 16 floats column-major (applied to centre, covariance and normal);
   *   tint: [r, g, b, strength 0..1] mixed into the colour before grading.
   */
  setInstance(label, fields = {}) {
    assertLabel(label);
    const base = label * INSTANCE_WORDS;
    if (fields.xform != null) {
      assertVec("xform", fields.xform, 16);
      this._instanceF32.set(fields.xform, base);
    }
    if (fields.tint != null) {
      assertVec("tint", fields.tint, 4);
      this._instanceF32.set(fields.tint, base + 16);
    }
    if (fields.visible != null) this._instanceU32[base + 20] = fields.visible ? 1 : 0;
    if (fields.selected != null) this._instanceU32[base + 21] = fields.selected ? 1 : 0;
    this._uploadInstance(label);
  }

  /** @returns {{xform: Float32Array, tint: number[], visible: boolean, selected: boolean}} */
  getInstance(label) {
    assertLabel(label);
    const base = label * INSTANCE_WORDS;
    return {
      xform: this._instanceF32.slice(base, base + 16),
      tint: Array.from(this._instanceF32.subarray(base + 16, base + 20)),
      visible: this._instanceU32[base + 20] !== 0,
      selected: this._instanceU32[base + 21] !== 0,
    };
  }

  /** Reset every instance record to identity / no tint / visible / unselected. */
  resetInstances() {
    for (let i = 0; i < MAX_INSTANCES; i++) this._resetInstanceRecord(i);
    this._idCache = null;
    this._stateVersion++;
    if (this.device && this.instanceBuffer) {
      this.device.queue.writeBuffer(this.instanceBuffer, 0, this._instanceData);
    }
  }

  // ---------------------------------------------------------------- camera

  setCamera(projection, view, focal, viewport, eye) {
    this.camera.projection.set(projection);
    this.camera.view.set(view);
    this.camera.focal = focal;
    this.camera.viewport = viewport;
    if (eye) this.camera.eye = eye;
    this._idCache = null;
    this._stateVersion++;
  }

  setParams(partial) {
    if (partial && partial.isolateLabel != null) {
      if (!isNonNegInt(partial.isolateLabel) || partial.isolateLabel >= MAX_INSTANCES) {
        throw new Error(`isolateLabel must be an integer in [0, ${MAX_INSTANCES}), got ${partial.isolateLabel}`);
      }
    }
    Object.assign(this.params, partial);
    this._idCache = null;
    this._stateVersion++;
  }

  _writeUniforms(mode = 0, focal = this.camera.focal, viewport = this.camera.viewport) {
    const u = this._uniformF32;
    const ui = this._uniformU32;
    u.set(this.camera.projection, 0);
    u.set(this.camera.view, 16);
    u[32] = focal[0];
    u[33] = focal[1];
    u[34] = viewport[0];
    u[35] = viewport[1];
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
    ui[56] = mode >>> 0;
    ui[57] = Math.min(Math.max(0, this.params.isolateLabel | 0), MAX_INSTANCES - 1) >>> 0;
    ui[58] = 0;
    ui[59] = 0;
    u[60] = this.params.selectionHighlight;
    u[61] = this.params.idAlphaThreshold;
    u[62] = 0;
    u[63] = 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this._uniformData);

    const sf = new Float32Array(this._sortUniformData);
    const sui = new Uint32Array(this._sortUniformData);
    sf.set(this.camera.view, 0);
    sui[16] = this.count;
    sui[17] = this.capacity;
    this.device.queue.writeBuffer(this.sortUniformBuffer, 0, this._sortUniformData);
  }

  // ------------------------------------------------------------- rendering

  /** Uniforms + histogram/minmax reset for one frame. */
  _prepareFrame(mode, focal, viewport) {
    this._writeUniforms(mode, focal, viewport);
    this.device.queue.writeBuffer(this.histBuffer, 0, this.zeroHist);
    this.device.queue.writeBuffer(this.minmaxBuffer, 0, this.minmaxInit);
  }

  /** Encode the 65536-bin counting sort (shared by canvas and offscreen paths). */
  _encodeSort(encoder) {
    const groups = Math.ceil(this.count / WG);
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
  }

  /** Encode one render pass drawing all gaussians into `view`. */
  _encodeDraw(encoder, { view, depthView = null, mode = 0, format, clearColor = [0, 0, 0, 0] }) {
    const isId = mode === OUTPUT_MODE.ID;
    const descriptor = {
      colorAttachments: [
        {
          view,
          clearValue: { r: clearColor[0], g: clearColor[1], b: clearColor[2], a: clearColor[3] },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    };
    if (isId) {
      descriptor.depthStencilAttachment = {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      };
    }
    const pass = encoder.beginRenderPass(descriptor);
    if (this.count > 0 && this.renderBindGroup) {
      pass.setPipeline(isId ? this.idPipeline : this._getBlendPipeline(format));
      pass.setBindGroup(0, this.renderBindGroup);
      pass.setVertexBuffer(0, this.quadBuffer);
      pass.draw(4, this.count);
    }
    pass.end();
  }

  /** Render colour to the canvas (no-op with a warning in offscreenOnly mode). */
  render(clearColor = [0, 0, 0, 0]) {
    if (!this.context) {
      if (!this._warnedNoContext) {
        console.warn("WebGPUSplatRenderer.render(): no canvas context (offscreenOnly); use renderOffscreen()");
        this._warnedNoContext = true;
      }
      return;
    }
    if (!this.device || this.count === 0 || !this.sortBindGroup) return;
    this._prepareFrame(OUTPUT_MODE.COLOR);
    const encoder = this.device.createCommandEncoder();
    this._encodeSort(encoder);
    this._encodeDraw(encoder, {
      view: this.context.getCurrentTexture().createView(),
      mode: OUTPUT_MODE.COLOR,
      format: this.format,
      clearColor,
    });
    this.device.queue.submit([encoder.finish()]);
  }

  // ------------------------------------------------------------- offscreen

  _serial(task) {
    const run = this._queue.then(task, task);
    this._queue = run.catch(() => {});
    return run;
  }

  _targetFormat(mode) {
    return TARGET_FORMATS[mode] || { format: this.floatFormat, bpp: this.floatFormat === "rgba32float" ? 16 : 8 };
  }

  /** Cached offscreen texture + readback buffer per mode/size. */
  _getTarget(mode, width, height) {
    const existing = this._targets.get(mode);
    if (existing && existing.width === width && existing.height === height) return existing;
    if (existing) {
      existing.texture.destroy();
      if (existing.depthTexture) existing.depthTexture.destroy();
      existing.readBuffer.destroy();
    }
    const { format, bpp } = this._targetFormat(mode);
    const bytesPerRow = align256(width * bpp);
    const texture = this.device.createTexture({
      size: [width, height],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depthTexture =
      mode === OUTPUT_MODE.ID
        ? this.device.createTexture({
            size: [width, height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null;
    const readBuffer = this.device.createBuffer({
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const target = {
      mode,
      width,
      height,
      format,
      bpp,
      bytesPerRow,
      texture,
      view: texture.createView(),
      depthTexture,
      depthView: depthTexture ? depthTexture.createView() : null,
      readBuffer,
    };
    this._targets.set(mode, target);
    return target;
  }

  /**
   * Composite a premultiplied RGBA8 layer over a solid background:
   * out = layer + clear·clearAlpha·(1 − layerAlpha). The GPU target is always
   * cleared to transparent because the front-to-back "under" blend would
   * otherwise treat a pre-filled alpha as an occluder in front of every splat.
   */
  _compositeClear(data, clearColor) {
    const ca = clearColor[3];
    if (!(ca > 0)) return;
    const bg = [clearColor[0] * ca * 255, clearColor[1] * ca * 255, clearColor[2] * ca * 255, ca * 255];
    for (let o = 0; o < data.length; o += 4) {
      const t = 1 - data[o + 3] / 255;
      if (t <= 0) continue;
      data[o] += bg[0] * t;
      data[o + 1] += bg[1] * t;
      data[o + 2] += bg[2] * t;
      data[o + 3] += bg[3] * t;
    }
  }

  _decodeReadback(target, mapped, clearColor = [0, 0, 0, 0]) {
    const { mode, width, height, bytesPerRow, format } = target;
    const n = width * height;
    if (mode === OUTPUT_MODE.COLOR) {
      const src = new Uint8Array(mapped);
      const data = new Uint8ClampedArray(n * 4);
      for (let y = 0; y < height; y++) {
        data.set(src.subarray(y * bytesPerRow, y * bytesPerRow + width * 4), y * width * 4);
      }
      this._compositeClear(data, clearColor);
      return { data };
    }
    if (mode === OUTPUT_MODE.ID) {
      const src = new Uint32Array(mapped);
      const data = new Uint32Array(n);
      const rowWords = bytesPerRow / 4;
      for (let y = 0; y < height; y++) {
        data.set(src.subarray(y * rowWords, y * rowWords + width), y * width);
      }
      return { data };
    }
    const isHalf = format === "rgba16float";
    const src = isHalf ? new Uint16Array(mapped) : new Float32Array(mapped);
    const rowElems = bytesPerRow / (isHalf ? 2 : 4);
    const read = (i) => (isHalf ? halfToFloat(src[i]) : src[i]);
    const alpha = new Float32Array(n);
    if (mode === OUTPUT_MODE.DEPTH) {
      const data = new Float32Array(n);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const s = y * rowElems + x * 4;
          const a = read(s + 3);
          alpha[y * width + x] = a;
          data[y * width + x] = a > 0 ? read(s) / a : 0;
        }
      }
      return { data, alpha };
    }
    const data = new Float32Array(n * 3);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const s = y * rowElems + x * 4;
        const a = read(s + 3);
        const p = y * width + x;
        alpha[p] = a;
        if (a > 0) {
          let nx = (read(s) / a) * 2 - 1;
          let ny = (read(s + 1) / a) * 2 - 1;
          let nz = (read(s + 2) / a) * 2 - 1;
          const len = Math.hypot(nx, ny, nz) || 1;
          data[p * 3] = nx / len;
          data[p * 3 + 1] = ny / len;
          data[p * 3 + 2] = nz / len;
        }
      }
    }
    return { data, alpha };
  }

  /**
   * Render one output mode to an offscreen target and read it back.
   *
   * The current camera is used; focal is rescaled by width/viewport so the
   * image is a resampled version of what the canvas shows. Modes:
   *   0 colour → data: Uint8ClampedArray RGBA, premultiplied by alpha (not
   *              un-premultiplied). The GPU target is always cleared to
   *              transparent; `clearColor` ([r, g, b, a] in 0..1, default
   *              transparent) is composited *behind* the splats on the CPU
   *              (out = layer + clear·a·(1 − layerAlpha)), so an opaque
   *              background such as [1, 1, 1, 1] gives a normal image instead
   *              of the fully attenuated result the under operator would produce.
   *   1 depth  → data: Float32Array expected depth per pixel (alpha-weighted mean of
   *              view distance -cam.z, i.e. the 2DGS "mean depth", depth_ratio = 0),
   *              already divided by accumulated alpha; 0 where alpha == 0; plus alpha.
   *              DEVIATION from plan §3.2.A, which specifies the 2DGS *median* depth
   *              (first sample with accumulated T < 0.5): a single hardware-blended
   *              pass cannot see the per-pixel prefix transmittance, so the mean is
   *              returned for now. Fine on single-surface pixels (F1 acceptance) but
   *              blends foreground/background at silhouettes; the median variant is
   *              to be produced by the K-buffer resolve (F3) before F6 TSDF fusion
   *              consumes it.
   *   2 normal → data: Float32Array(w*h*3) unit view-space normals (smallest-scale axis
   *              facing the camera, +z towards the camera), zeros where alpha == 0; plus alpha
   *   3 id     → data: Uint32Array gaussian index + 1 (0 = nothing) of the first gaussian
   *              with alpha ≥ params.idAlphaThreshold (depth-tested, no blending)
   * Depth/normal use rgba32float when 'float32-blendable' is available, else rgba16float
   * (see this.floatFormat). Calls are serialised; targets are cached per mode/size.
   *
   * @param {{mode?: number, width: number, height: number, clearColor?: number[]}} opts
   *   clearColor: 4 finite numbers (clamped to 0..1); only used in colour mode
   * @returns {Promise<{mode:number,width:number,height:number,data:ArrayBufferView,alpha?:Float32Array,
   *   format:string,camera:{projection:Float32Array,view:Float32Array,focal:number[],viewport:number[],eye:number[]}}>}
   */
  renderOffscreen({ mode = OUTPUT_MODE.COLOR, width, height, clearColor = [0, 0, 0, 0] } = {}) {
    if (!this.device) return Promise.reject(new Error("WebGPU is not initialized"));
    if (!Object.values(OUTPUT_MODE).includes(mode)) {
      return Promise.reject(new Error(`unknown output mode ${mode}`));
    }
    const maxDim = this.device.limits.maxTextureDimension2D || 8192;
    if (!isNonNegInt(width) || !isNonNegInt(height) || width < 1 || height < 1 || width > maxDim || height > maxDim) {
      return Promise.reject(new Error(`invalid offscreen size ${width}x${height}`));
    }
    let clear;
    try {
      assertVec("clearColor", clearColor, 4);
      clear = Array.from(clearColor, (c) => Math.min(1, Math.max(0, c)));
    } catch (err) {
      return Promise.reject(err);
    }
    return this._serial(() => this._renderOffscreenNow(mode, width, height, clear));
  }

  async _renderOffscreenNow(mode, width, height, clearColor) {
    const target = this._getTarget(mode, width, height);
    const vp = this.camera.viewport;
    const focal = [
      this.camera.focal[0] * (width / Math.max(vp[0], 1e-6)),
      this.camera.focal[1] * (height / Math.max(vp[1], 1e-6)),
    ];
    const viewport = [width, height];
    this._prepareFrame(mode, focal, viewport);
    const encoder = this.device.createCommandEncoder();
    if (this.count > 0 && this.sortBindGroup) this._encodeSort(encoder);
    // Always clear to transparent: with "under" blending a pre-filled alpha
    // would occlude every splat. The requested clearColor is composited on the CPU.
    this._encodeDraw(encoder, {
      view: target.view,
      depthView: target.depthView,
      mode,
      format: target.format,
      clearColor: [0, 0, 0, 0],
    });
    encoder.copyTextureToBuffer(
      { texture: target.texture },
      { buffer: target.readBuffer, bytesPerRow: target.bytesPerRow },
      { width, height }
    );
    this.device.queue.submit([encoder.finish()]);
    await target.readBuffer.mapAsync(GPUMapMode.READ);
    let decoded;
    try {
      decoded = this._decodeReadback(
        target,
        target.readBuffer.getMappedRange(),
        mode === OUTPUT_MODE.COLOR ? clearColor : undefined
      );
    } finally {
      target.readBuffer.unmap();
    }
    return {
      mode,
      width,
      height,
      format: target.format,
      ...decoded,
      camera: {
        projection: this.camera.projection.slice(),
        view: this.camera.view.slice(),
        focal,
        viewport,
        eye: Array.from(this.camera.eye),
      },
    };
  }

  _viewportSize() {
    return [
      Math.max(1, Math.round(this.camera.viewport[0])),
      Math.max(1, Math.round(this.camera.viewport[1])),
    ];
  }

  /** ID readback at the camera viewport size, cached until any state changes. */
  async _idFrame() {
    const [w, h] = this._viewportSize();
    const c = this._idCache;
    if (c && c.version === this._stateVersion && c.width === w && c.height === h) return c;
    const version = this._stateVersion;
    const res = await this.renderOffscreen({ mode: OUTPUT_MODE.ID, width: w, height: h });
    const frame = { version, width: w, height: h, data: res.data };
    if (this._stateVersion === version) this._idCache = frame;
    return frame;
  }

  /**
   * Gaussian under a canvas pixel (device pixels, origin top-left).
   * @param {number} x
   * @param {number} y
   * @param {{depth?: boolean}} [opts] depth: also read the expected depth at that pixel
   * @returns {Promise<{index:number, label:number, group:number, depth:number|null}>} index -1 = nothing; group = superpoint id + 1 (0 = sin grupo)
   */
  async pick(x, y, opts = {}) {
    const frame = await this._idFrame();
    const px = Math.floor(x);
    const py = Math.floor(y);
    let index = -1;
    if (px >= 0 && py >= 0 && px < frame.width && py < frame.height) {
      index = frame.data[py * frame.width + px] - 1;
    }
    const label = index >= 0 && index < this._labels.length ? this._labels[index] : 0;
    const group = index >= 0 && index < this._groups.length ? this._groups[index] : 0;
    let depth = null;
    if (opts && opts.depth && index >= 0) {
      const d = await this.renderOffscreen({ mode: OUTPUT_MODE.DEPTH, width: frame.width, height: frame.height });
      const p = py * frame.width + px;
      depth = d.alpha[p] > 0 ? d.data[p] : null;
    }
    return { index, label, group, depth };
  }

  /**
   * Unique gaussian indices visible (per the ID pass) inside a pixel rectangle.
   * @returns {Promise<Uint32Array>} ascending indices
   */
  async pickRect(x0, y0, x1, y1) {
    const frame = await this._idFrame();
    const xa = Math.max(0, Math.floor(Math.min(x0, x1)));
    const ya = Math.max(0, Math.floor(Math.min(y0, y1)));
    const xb = Math.min(frame.width - 1, Math.floor(Math.max(x0, x1)));
    const yb = Math.min(frame.height - 1, Math.floor(Math.max(y0, y1)));
    const seen = new Set();
    for (let y = ya; y <= yb; y++) {
      for (let x = xa; x <= xb; x++) {
        const v = frame.data[y * frame.width + x];
        if (v) seen.add(v - 1);
      }
    }
    return Uint32Array.from(seen).sort();
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
