/**
 * Plan F5 — encoders for per-instance / scene export from the 12-float cloud
 * used by the viewer (see shared/splat-io.js: xyz, opacity, scale xyz, pad,
 * quat wxyz; SH tightly packed, 48 floats).
 *
 *   encodeSplat32(cloud)          → .splat (32-byte rows, SH0 only), readable by every viewer here
 *   encodePly(cloud, extras)      → binary_little_endian 3DGS PLY with f_dc/f_rest/opacity/scale/rot
 *                                   plus the plan's extra columns: instance_id, class_id, confidence
 * Both round-trip through toGaussianCloud() (tests/unit/export-io.test.mjs).
 * SPZ / compressed PLY are produced by GaussForge from the PLY (parse-worker "convert").
 */

const SH_C0 = 0.28209479177387814;
const REST_COUNT = { 0: 0, 1: 3, 2: 8, 3: 15 };

function clampByte(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function logit(p) {
  const q = Math.min(1 - 1e-6, Math.max(1e-6, p));
  return Math.log(q / (1 - q));
}

/** {gaussians, sh, count} → Uint8Array of count*32 bytes. */
export function encodeSplat32({ gaussians, sh = null, count = gaussians.length / 12 }) {
  const out = new Uint8Array(count * 32);
  const f32 = new Float32Array(out.buffer);
  for (let i = 0; i < count; i++) {
    const g = i * 12;
    const f = i * 8;
    const b = i * 32;
    f32[f] = gaussians[g];
    f32[f + 1] = gaussians[g + 1];
    f32[f + 2] = gaussians[g + 2];
    f32[f + 3] = gaussians[g + 4];
    f32[f + 4] = gaussians[g + 5];
    f32[f + 5] = gaussians[g + 6];
    const s = i * 48;
    const r = sh ? 0.5 + SH_C0 * sh[s] : 0.5;
    const gg = sh ? 0.5 + SH_C0 * sh[s + 1] : 0.5;
    const bb = sh ? 0.5 + SH_C0 * sh[s + 2] : 0.5;
    out[b + 24] = clampByte(r * 255);
    out[b + 25] = clampByte(gg * 255);
    out[b + 26] = clampByte(bb * 255);
    out[b + 27] = clampByte(gaussians[g + 3] * 255);
    const qw = gaussians[g + 8], qx = gaussians[g + 9], qy = gaussians[g + 10], qz = gaussians[g + 11];
    const qn = Math.hypot(qw, qx, qy, qz) || 1;
    out[b + 28] = clampByte((qw / qn) * 128 + 128);
    out[b + 29] = clampByte((qx / qn) * 128 + 128);
    out[b + 30] = clampByte((qy / qn) * 128 + 128);
    out[b + 31] = clampByte((qz / qn) * 128 + 128);
  }
  return out;
}

/**
 * 3DGS PLY (binary little-endian). `extras.labels` (Uint32Array) adds
 * `instance_id`, `extras.classIds` adds `class_id` and `extras.confidences`
 * adds `confidence`, one value per gaussian, so a reload can rebuild the
 * instances (plan §3.2.E "PLY propio con propiedades extra"). The extra columns
 * are stored as `float` (exact for ids < 2^24): GaussForge and most 3DGS tools
 * only accept float vertex properties; readPlyColumns() returns the ids as u32.
 * @returns {ArrayBuffer}
 */
export function encodePly({ gaussians, sh = null, shDegree = 0, count = gaussians.length / 12 }, extras = {}) {
  const nRest = REST_COUNT[Math.max(0, Math.min(3, shDegree | 0))];
  const props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"];
  for (let k = 0; k < nRest * 3; k++) props.push(`f_rest_${k}`);
  props.push("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3");
  const extraCols = [];
  if (extras.labels) {
    if (extras.labels.length !== count) throw new Error("labels debe tener un valor por gaussiana");
    extraCols.push({ name: "instance_id", type: "float", values: extras.labels });
  }
  if (extras.classIds) {
    if (extras.classIds.length !== count) throw new Error("classIds debe tener un valor por gaussiana");
    extraCols.push({ name: "class_id", type: "float", values: extras.classIds });
  }
  if (extras.confidences) {
    if (extras.confidences.length !== count) throw new Error("confidences debe tener un valor por gaussiana");
    extraCols.push({ name: "confidence", type: "float", values: extras.confidences });
  }
  const header =
    "ply\nformat binary_little_endian 1.0\n" +
    (extras.comment ? `comment ${String(extras.comment).replace(/[\r\n]+/g, " ")}\n` : "") +
    `element vertex ${count}\n` +
    props.map((p) => `property float ${p}\n`).join("") +
    extraCols.map((c) => `property ${c.type} ${c.name}\n`).join("") +
    "end_header\n";
  const headerBytes = new TextEncoder().encode(header);
  const rowFloats = props.length + extraCols.length;
  const rowBytes = rowFloats * 4;
  const out = new ArrayBuffer(headerBytes.length + count * rowBytes);
  new Uint8Array(out).set(headerBytes);
  const view = new DataView(out, headerBytes.length);
  for (let i = 0; i < count; i++) {
    const g = i * 12;
    const s = i * 48;
    let o = i * rowBytes;
    const put = (v) => { view.setFloat32(o, v, true); o += 4; };
    put(gaussians[g]); put(gaussians[g + 1]); put(gaussians[g + 2]);
    put(0); put(0); put(0);
    put(sh ? sh[s] : 0); put(sh ? sh[s + 1] : 0); put(sh ? sh[s + 2] : 0);
    // f_rest is stored colour-major: all coefficients of R, then G, then B.
    for (let c = 0; c < 3; c++) for (let k = 0; k < nRest; k++) put(sh ? sh[s + 3 + k * 3 + c] : 0);
    put(logit(gaussians[g + 3]));
    put(Math.log(Math.max(1e-9, gaussians[g + 4])));
    put(Math.log(Math.max(1e-9, gaussians[g + 5])));
    put(Math.log(Math.max(1e-9, gaussians[g + 6])));
    put(gaussians[g + 8]); put(gaussians[g + 9]); put(gaussians[g + 10]); put(gaussians[g + 11]);
    for (const c of extraCols) {
      view.setFloat32(o, c.values[i], true);
      o += 4;
    }
  }
  return out;
}

/** Suggested file name for an export: <escena>/<id_instancia|escena>.<ext>. */
export function exportFileName(escena, label, format) {
  const base = String(escena || "escena").replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "escena";
  const ext = format === "compressed.ply" ? "compressed.ply" : format;
  return `${base}_${label != null ? `instancia-${label}` : "escena"}.${ext}`;
}
