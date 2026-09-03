/**
 * Plan F5 — object editing as a reproducible operations log (`ops.jsonl`).
 *
 * An EditSession owns a copy of the cloud (gaussians, SH), the per-gaussian
 * instance labels, the invariant `origen` (original gaussian index, plan §3.3)
 * and per-instance state (rigid/affine transform, deleted flag, name). Every
 * change is an op (plain JSON object, Spanish schema); `replay(base, ops)`
 * rebuilds the same session deterministically, which gives undo/redo and the
 * acceptance criterion "reproducir ops.jsonl da el mismo resultado".
 *
 * Ops (one JSON object per line in ops.jsonl):
 *   {"op":"asignar",     "id_instancia":3, "rangos":[[i0,i1],...]}   gaussian ranges → label (0 = fondo)
 *   {"op":"transformar", "id_instancia":3, "xform":[16 floats column-major]}   absolute transform
 *   {"op":"borrar",      "id_instancia":3}                             hidden and excluded from export
 *   {"op":"restaurar",   "id_instancia":3}                             undo a borrar
 *   {"op":"duplicar",    "id_instancia":3, "nueva":7, "xform":[16]?}  copies the gaussians under a new label
 *   {"op":"fusionar",    "origen":7, "destino":3}                     relabel origen → destino
 *   {"op":"renombrar",   "id_instancia":3, "nombre_es":"silla"}
 * Every op may carry "fecha" (ISO) and "nota"; replay ignores them.
 *
 * Matrices are column-major 4×4 (same convention as gpu-renderer.js xform).
 */

export const GAUSSIAN_STRIDE = 12;
export const SH_STRIDE = 48;
export const OP_NAMES = Object.freeze(["asignar", "transformar", "borrar", "restaurar", "duplicar", "fusionar", "renombrar"]);

// ---------------------------------------------------------------- matrices

export function mat4Identity() {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
}

/** a * b, column-major. */
export function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function mat4Translation([x, y, z]) {
  const m = mat4Identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

export function mat4Scaling(s) {
  const v = Array.isArray(s) || ArrayBuffer.isView(s) ? s : [s, s, s];
  const m = mat4Identity();
  m[0] = v[0];
  m[5] = v[1];
  m[10] = v[2];
  return m;
}

/** Rotation of `deg` degrees about a unit axis (Rodrigues). */
export function mat4RotationAxis(axis, deg) {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const x = axis[0] / n, y = axis[1] / n, z = axis[2] / n;
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t), s = Math.sin(t), k = 1 - c;
  const m = mat4Identity();
  m[0] = c + x * x * k;  m[4] = x * y * k - z * s; m[8] = x * z * k + y * s;
  m[1] = y * x * k + z * s; m[5] = c + y * y * k;  m[9] = y * z * k - x * s;
  m[2] = z * x * k - y * s; m[6] = z * y * k + x * s; m[10] = c + z * z * k;
  return m;
}

/**
 * T(pivot) · T(translate) · R(axis, deg) · S(scale) · T(-pivot): the rotation and
 * scale happen about `pivot` (instance centre), then the translation is added.
 */
export function composeTransform({ translate = [0, 0, 0], rotateAxis = [0, 1, 0], rotateDeg = 0, scale = 1, pivot = [0, 0, 0] } = {}) {
  let m = mat4Translation([-pivot[0], -pivot[1], -pivot[2]]);
  m = mat4Multiply(mat4Scaling(scale), m);
  if (rotateDeg) m = mat4Multiply(mat4RotationAxis(rotateAxis, rotateDeg), m);
  m = mat4Multiply(mat4Translation(translate), m);
  m = mat4Multiply(mat4Translation(pivot), m);
  return m;
}

export function transformPoint(m, p) {
  const x = p[0], y = p[1], z = p[2];
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

export function isIdentity(m, eps = 1e-7) {
  const I = mat4Identity();
  for (let i = 0; i < 16; i++) if (Math.abs(m[i] - I[i]) > eps) return false;
  return true;
}

/** Rotation quaternion [w,x,y,z] of the 3×3 block after removing per-column scale. */
export function rotationQuatFromMat4(m) {
  const sx = Math.hypot(m[0], m[1], m[2]) || 1;
  const sy = Math.hypot(m[4], m[5], m[6]) || 1;
  const sz = Math.hypot(m[8], m[9], m[10]) || 1;
  const r00 = m[0] / sx, r10 = m[1] / sx, r20 = m[2] / sx;
  const r01 = m[4] / sy, r11 = m[5] / sy, r21 = m[6] / sy;
  const r02 = m[8] / sz, r12 = m[9] / sz, r22 = m[10] / sz;
  const tr = r00 + r11 + r22;
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (r21 - r12) / s; y = (r02 - r20) / s; z = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const s = Math.sqrt(1 + r00 - r11 - r22) * 2;
    w = (r21 - r12) / s; x = 0.25 * s; y = (r01 + r10) / s; z = (r02 + r20) / s;
  } else if (r11 > r22) {
    const s = Math.sqrt(1 + r11 - r00 - r22) * 2;
    w = (r02 - r20) / s; x = (r01 + r10) / s; y = 0.25 * s; z = (r12 + r21) / s;
  } else {
    const s = Math.sqrt(1 + r22 - r00 - r11) * 2;
    w = (r10 - r01) / s; x = (r02 + r20) / s; y = (r12 + r21) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z) || 1;
  return { quat: [w / n, x / n, y / n, z / n], scale: [sx, sy, sz] };
}

function quatMultiply(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

// ---------------------------------------------------------------- index ranges

/** Sorted unique indices → [[start, end], ...] inclusive ranges (compact JSON). */
export function rangesFromIndices(indices) {
  const sorted = Array.from(indices, (v) => v | 0).sort((a, b) => a - b);
  const out = [];
  for (const v of sorted) {
    const last = out[out.length - 1];
    if (last && v === last[1]) continue;
    if (last && v === last[1] + 1) last[1] = v;
    else out.push([v, v]);
  }
  return out;
}

export function indicesFromRanges(ranges) {
  let n = 0;
  for (const [a, b] of ranges) n += b - a + 1;
  const out = new Uint32Array(n);
  let k = 0;
  for (const [a, b] of ranges) for (let v = a; v <= b; v++) out[k++] = v;
  return out;
}

// ---------------------------------------------------------------- session

/**
 * Editable copy of a cloud plus the per-instance state. `base` arrays are
 * copied, never mutated; `labels` may be omitted (all fondo).
 */
export class EditSession {
  constructor({ gaussians, sh = null, shDegree = 0, labels = null, names = {} }) {
    if (!(gaussians instanceof Float32Array) || gaussians.length % GAUSSIAN_STRIDE) throw new Error("gaussians: Float32Array de 12 floats por gaussiana");
    const count = gaussians.length / GAUSSIAN_STRIDE;
    if (labels && labels.length !== count) throw new Error("labels debe tener una etiqueta por gaussiana");
    this.count = count;
    this.shDegree = shDegree | 0;
    this.gaussians = gaussians.slice();
    this.sh = sh && sh.length ? sh.slice() : new Float32Array(count * SH_STRIDE);
    this.labels = labels ? Uint32Array.from(labels) : new Uint32Array(count);
    /** Original gaussian index of every gaussian (duplicates point at their source). */
    this.origen = new Uint32Array(count);
    for (let i = 0; i < count; i++) this.origen[i] = i;
    /** @type {Map<number, Float32Array>} absolute transform per label (absent = identity) */
    this.xforms = new Map();
    /** @type {Set<number>} */
    this.deleted = new Set();
    /** @type {Map<number, string>} */
    this.names = new Map(Object.entries(names).map(([k, v]) => [Number(k), String(v)]));
  }

  /** Labels in use (non-zero), ascending. */
  labelSet() {
    const s = new Set();
    for (let i = 0; i < this.labels.length; i++) if (this.labels[i]) s.add(this.labels[i]);
    return [...s].sort((a, b) => a - b);
  }

  nextLabel() {
    let max = 0;
    for (let i = 0; i < this.labels.length; i++) if (this.labels[i] > max) max = this.labels[i];
    for (const l of this.xforms.keys()) if (l > max) max = l;
    return max + 1;
  }

  countOf(label) {
    let n = 0;
    for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === label) n++;
    return n;
  }

  indicesOf(label) {
    const out = [];
    for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === label) out.push(i);
    return Uint32Array.from(out);
  }

  xformOf(label) {
    return this.xforms.get(label) || mat4Identity();
  }

  /** Axis-aligned centre of a label's gaussian centres (untransformed). */
  centreOf(label) {
    let n = 0;
    const c = [0, 0, 0];
    for (let i = 0; i < this.labels.length; i++) {
      if (this.labels[i] !== label) continue;
      c[0] += this.gaussians[i * 12];
      c[1] += this.gaussians[i * 12 + 1];
      c[2] += this.gaussians[i * 12 + 2];
      n++;
    }
    return n ? [c[0] / n, c[1] / n, c[2] / n] : null;
  }

  _appendCopies(indices) {
    const n = indices.length;
    const g = new Float32Array((this.count + n) * GAUSSIAN_STRIDE);
    const s = new Float32Array((this.count + n) * SH_STRIDE);
    const l = new Uint32Array(this.count + n);
    const o = new Uint32Array(this.count + n);
    g.set(this.gaussians);
    s.set(this.sh);
    l.set(this.labels);
    o.set(this.origen);
    for (let k = 0; k < n; k++) {
      const src = indices[k];
      const dst = this.count + k;
      g.set(this.gaussians.subarray(src * 12, src * 12 + 12), dst * 12);
      s.set(this.sh.subarray(src * 48, src * 48 + 48), dst * 48);
      o[dst] = this.origen[src];
    }
    this.gaussians = g;
    this.sh = s;
    this.labels = l;
    this.origen = o;
    this.count += n;
    return this.count - n;
  }

  /** Apply one op in place. Throws on malformed ops (nothing is changed then). */
  apply(op) {
    if (!op || typeof op !== "object") throw new Error("op inválida");
    const label = (v) => {
      if (!Number.isInteger(v) || v < 0) throw new Error(`id_instancia inválido: ${v}`);
      return v;
    };
    switch (op.op) {
      case "asignar": {
        const target = label(op.id_instancia);
        const idx = op.rangos ? indicesFromRanges(op.rangos) : Uint32Array.from(op.indices || []);
        for (const i of idx) {
          if (i >= this.count) throw new Error(`índice fuera de rango: ${i}`);
        }
        for (const i of idx) this.labels[i] = target;
        return { count: idx.length };
      }
      case "transformar": {
        const l = label(op.id_instancia);
        if (!op.xform || op.xform.length !== 16) throw new Error("xform debe tener 16 valores");
        const m = Float32Array.from(op.xform);
        if (isIdentity(m)) this.xforms.delete(l);
        else this.xforms.set(l, m);
        return { label: l };
      }
      case "borrar":
        this.deleted.add(label(op.id_instancia));
        return { label: op.id_instancia };
      case "restaurar":
        this.deleted.delete(label(op.id_instancia));
        return { label: op.id_instancia };
      case "duplicar": {
        const src = label(op.id_instancia);
        const nueva = label(op.nueva);
        if (nueva === 0) throw new Error("nueva no puede ser 0 (fondo)");
        const idx = this.indicesOf(src);
        if (!idx.length) throw new Error(`la instancia ${src} no tiene gaussianas`);
        const start = this._appendCopies(idx);
        for (let k = 0; k < idx.length; k++) this.labels[start + k] = nueva;
        const m = op.xform ? Float32Array.from(op.xform) : this.xformOf(src);
        if (isIdentity(m)) this.xforms.delete(nueva);
        else this.xforms.set(nueva, m);
        this.names.set(nueva, op.nombre_es || `${this.names.get(src) || `instancia ${src}`} (copia)`);
        return { label: nueva, count: idx.length, start };
      }
      case "fusionar": {
        const from = label(op.origen);
        const to = label(op.destino);
        let n = 0;
        for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === from) { this.labels[i] = to; n++; }
        this.xforms.delete(from);
        this.deleted.delete(from);
        this.names.delete(from);
        return { count: n };
      }
      case "renombrar":
        this.names.set(label(op.id_instancia), String(op.nombre_es || ""));
        return { label: op.id_instancia };
      default:
        throw new Error(`op desconocida: ${op.op}`);
    }
  }
}

/** Fresh session from `base` with every op applied in order. */
export function replay(base, ops) {
  const s = new EditSession(base);
  for (const op of ops) s.apply(op);
  return s;
}

export function opsToJsonl(ops) {
  return ops.map((op) => JSON.stringify(op)).join("\n") + (ops.length ? "\n" : "");
}

export function opsFromJsonl(text) {
  const ops = [];
  for (const line of String(text).split(/\r?\n/)) {
    const t = line.trim();
    if (!t) continue;
    const op = JSON.parse(t);
    if (!OP_NAMES.includes(op.op)) throw new Error(`op desconocida en ops.jsonl: ${op.op}`);
    ops.push(op);
  }
  return ops;
}

/**
 * Undo/redo on top of replay: `log.push(op)` applies it, `undo()` rebuilds the
 * session from the base without the last op, `redo()` re-applies it.
 */
export class EditLog {
  constructor(base, ops = []) {
    this.base = base;
    this.ops = [];
    this.undone = [];
    this.session = new EditSession(base);
    for (const op of ops) this.push(op);
  }

  push(op) {
    const stamped = { ...op, fecha: op.fecha || new Date().toISOString() };
    const result = this.session.apply(stamped);
    this.ops.push(stamped);
    this.undone = [];
    return result;
  }

  undo() {
    if (!this.ops.length) return null;
    const op = this.ops.pop();
    this.undone.push(op);
    this.session = replay(this.base, this.ops);
    return op;
  }

  redo() {
    if (!this.undone.length) return null;
    const op = this.undone.pop();
    this.session.apply(op);
    this.ops.push(op);
    return op;
  }

  toJsonl() {
    return opsToJsonl(this.ops);
  }
}

// ---------------------------------------------------------------- baking / subsets

/**
 * Apply an instance transform to one gaussian in place: centre, rotation
 * (quaternion) and per-axis scale. Exact for rigid + uniform scale; for
 * non-uniform scale the local axes keep their orientation and the column
 * norms scale the extents (an approximation when the covariance is not
 * aligned with the transform axes).
 */
export function transformGaussian(gaussians, i, m, decomposed = rotationQuatFromMat4(m)) {
  const g = i * GAUSSIAN_STRIDE;
  const p = transformPoint(m, [gaussians[g], gaussians[g + 1], gaussians[g + 2]]);
  gaussians[g] = p[0];
  gaussians[g + 1] = p[1];
  gaussians[g + 2] = p[2];
  const { quat, scale } = decomposed;
  const uniform = Math.abs(scale[0] - scale[1]) < 1e-6 && Math.abs(scale[1] - scale[2]) < 1e-6;
  if (uniform) {
    gaussians[g + 4] *= scale[0];
    gaussians[g + 5] *= scale[0];
    gaussians[g + 6] *= scale[0];
  } else {
    // Scale each local axis by the length of its image under the linear part.
    const q = [gaussians[g + 8], gaussians[g + 9], gaussians[g + 10], gaussians[g + 11]];
    const axes = quatAxes(q);
    for (let a = 0; a < 3; a++) {
      const ax = axes[a];
      const vx = m[0] * ax[0] + m[4] * ax[1] + m[8] * ax[2];
      const vy = m[1] * ax[0] + m[5] * ax[1] + m[9] * ax[2];
      const vz = m[2] * ax[0] + m[6] * ax[1] + m[10] * ax[2];
      gaussians[g + 4 + a] *= Math.hypot(vx, vy, vz);
    }
  }
  const r = quatMultiply(quat, [gaussians[g + 8], gaussians[g + 9], gaussians[g + 10], gaussians[g + 11]]);
  const n = Math.hypot(r[0], r[1], r[2], r[3]) || 1;
  gaussians[g + 8] = r[0] / n;
  gaussians[g + 9] = r[1] / n;
  gaussians[g + 10] = r[2] / n;
  gaussians[g + 11] = r[3] / n;
}

function quatAxes([w, x, y, z]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/**
 * Bake the session into an exportable cloud: transforms applied to the
 * gaussians, deleted instances dropped, optionally one instance only.
 * @param {EditSession} session
 * @param {{label?: number|null, includeBackground?: boolean, hidden?: Set<number>}} options
 *   label: export only this instance; includeBackground: keep label 0 when exporting the scene
 * @returns {{gaussians: Float32Array, sh: Float32Array, labels: Uint32Array, origen: Uint32Array, count: number, shDegree: number}}
 */
export function bakeSession(session, { label = null, includeBackground = true, hidden = null } = {}) {
  const keep = [];
  for (let i = 0; i < session.count; i++) {
    const l = session.labels[i];
    if (label != null && l !== label) continue;
    if (label == null && !includeBackground && l === 0) continue;
    if (session.deleted.has(l)) continue;
    if (hidden && hidden.has(l)) continue;
    keep.push(i);
  }
  const n = keep.length;
  const gaussians = new Float32Array(n * GAUSSIAN_STRIDE);
  const sh = new Float32Array(n * SH_STRIDE);
  const labels = new Uint32Array(n);
  const origen = new Uint32Array(n);
  const decomposed = new Map();
  for (let k = 0; k < n; k++) {
    const i = keep[k];
    gaussians.set(session.gaussians.subarray(i * 12, i * 12 + 12), k * 12);
    sh.set(session.sh.subarray(i * 48, i * 48 + 48), k * 48);
    labels[k] = session.labels[i];
    origen[k] = session.origen[i];
    const m = session.xforms.get(labels[k]);
    if (m) {
      if (!decomposed.has(labels[k])) decomposed.set(labels[k], rotationQuatFromMat4(m));
      transformGaussian(gaussians, k, m, decomposed.get(labels[k]));
    }
  }
  return { gaussians, sh, labels, origen, count: n, shDegree: session.shDegree };
}

/** FNV-1a over labels + xforms: a cheap fingerprint to compare replays. */
export function sessionFingerprint(session) {
  let h = 0x811c9dc5;
  const mix = (v) => {
    h ^= v & 0xff; h = Math.imul(h, 0x01000193);
    h ^= (v >>> 8) & 0xff; h = Math.imul(h, 0x01000193);
    h ^= (v >>> 16) & 0xff; h = Math.imul(h, 0x01000193);
    h ^= (v >>> 24) & 0xff; h = Math.imul(h, 0x01000193);
  };
  mix(session.count);
  for (let i = 0; i < session.labels.length; i++) mix(session.labels[i]);
  for (const l of [...session.xforms.keys()].sort((a, b) => a - b)) {
    mix(l);
    const m = session.xforms.get(l);
    for (let i = 0; i < 16; i++) mix(Math.round(m[i] * 1e6));
  }
  for (const l of [...session.deleted].sort((a, b) => a - b)) mix(l | 0x80000000);
  return (h >>> 0).toString(16).padStart(8, "0");
}
