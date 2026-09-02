/**
 * CPU-side instance/label bookkeeping for a gaussian cloud (no WebGPU, no deps).
 *
 * Invariant (plan §3.3): the gaussian index is the key; `labels[i]` is the
 * instance id of gaussian i, 0 = fondo (background). Label values mirror the
 * renderer's label buffer (see gpu-renderer.js setLabels/setLabel).
 *
 * Gaussian layout: Float32Array N*12 = [x,y,z, opacity, sx,sy,sz, pad, qw,qx,qy,qz].
 */

const GAUSSIAN_FLOATS = 12;
const GOLDEN_RATIO_CONJUGATE = 0.618033988749895;

function assertCount(count) {
  if (!Number.isInteger(count) || count < 0) {
    throw new Error(`count must be a non-negative integer, got ${count}`);
  }
}

function assertLabel(label) {
  if (!Number.isInteger(label) || label < 0 || label > 0xffffffff) {
    throw new Error(`label must be a non-negative integer, got ${label}`);
  }
}

/** HSV (h in [0,1)) → [r, g, b] in [0,1]. */
function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

/**
 * Deterministic palette: golden-ratio hue walk so consecutive labels are far
 * apart in hue. Label 0 (fondo) is a neutral grey.
 * @param {number} label
 * @returns {number[]} [r, g, b] in [0, 1]
 */
export function labelColor(label) {
  assertLabel(label);
  if (label === 0) return [0.6, 0.6, 0.6];
  const hue = (label * GOLDEN_RATIO_CONJUGATE) % 1;
  const sat = 0.62 + 0.18 * ((label * 7) % 3) / 2;
  const val = 0.95 - 0.15 * ((label * 5) % 2);
  return hsvToRgb(hue, sat, val);
}

export class InstanceSet {
  /**
   * @param {number} count number of gaussians
   */
  constructor(count) {
    assertCount(count);
    /** @type {Uint32Array} label per gaussian (0 = fondo) */
    this.labels = new Uint32Array(count);
  }

  /** Number of gaussians. */
  get count() {
    return this.labels.length;
  }

  /**
   * Assign `label` to every index in `indices`.
   * @param {ArrayLike<number>} indices
   * @param {number} label
   */
  assign(indices, label) {
    assertLabel(label);
    const n = this.labels.length;
    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      if (!Number.isInteger(i) || i < 0 || i >= n) {
        throw new Error(`gaussian index ${i} out of range [0, ${n})`);
      }
      this.labels[i] = label;
    }
  }

  /** @returns {Map<number, number>} label → number of gaussians */
  countByLabel() {
    const counts = new Map();
    for (let i = 0; i < this.labels.length; i++) {
      const l = this.labels[i];
      counts.set(l, (counts.get(l) || 0) + 1);
    }
    return counts;
  }

  /** @returns {Uint32Array} ascending indices of gaussians carrying `label` */
  indicesOf(label) {
    assertLabel(label);
    let n = 0;
    for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === label) n++;
    const out = new Uint32Array(n);
    let k = 0;
    for (let i = 0; i < this.labels.length; i++) if (this.labels[i] === label) out[k++] = i;
    return out;
  }

  /**
   * Axis-aligned bounds of the gaussian centres carrying `label`.
   * @param {number} label
   * @param {Float32Array} gaussians N*12 floats
   * @returns {{min:number[], max:number[], center:number[], count:number}|null} null when empty
   */
  boundsOf(label, gaussians) {
    assertLabel(label);
    if (!gaussians || gaussians.length !== this.labels.length * GAUSSIAN_FLOATS) {
      throw new Error(`gaussians must have ${GAUSSIAN_FLOATS} floats per gaussian (${this.labels.length})`);
    }
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    let count = 0;
    for (let i = 0; i < this.labels.length; i++) {
      if (this.labels[i] !== label) continue;
      const o = i * GAUSSIAN_FLOATS;
      for (let a = 0; a < 3; a++) {
        const v = gaussians[o + a];
        if (v < min[a]) min[a] = v;
        if (v > max[a]) max[a] = v;
      }
      count++;
    }
    if (count === 0) return null;
    const center = [0, 1, 2].map((a) => 0.5 * (min[a] + max[a]));
    return { min, max, center, count };
  }

  /**
   * Run-length encoded JSON form: { version, count, runs: [[label, length], ...] }.
   */
  toJSON() {
    const runs = [];
    const n = this.labels.length;
    let i = 0;
    while (i < n) {
      const l = this.labels[i];
      let j = i + 1;
      while (j < n && this.labels[j] === l) j++;
      runs.push([l, j - i]);
      i = j;
    }
    return { version: 1, count: n, runs };
  }

  /**
   * Inverse of toJSON(); also accepts { count, labels: number[] }.
   * @param {object|string} json
   * @returns {InstanceSet}
   */
  static fromJSON(json) {
    const obj = typeof json === "string" ? JSON.parse(json) : json;
    if (!obj || typeof obj !== "object") throw new Error("InstanceSet.fromJSON: invalid input");
    assertCount(obj.count);
    const set = new InstanceSet(obj.count);
    if (Array.isArray(obj.runs)) {
      let i = 0;
      for (const run of obj.runs) {
        if (!Array.isArray(run) || run.length !== 2) throw new Error("InstanceSet.fromJSON: bad run");
        const [label, len] = run;
        assertLabel(label);
        if (!Number.isInteger(len) || len < 0 || i + len > obj.count) {
          throw new Error("InstanceSet.fromJSON: runs exceed count");
        }
        set.labels.fill(label, i, i + len);
        i += len;
      }
      if (i !== obj.count) throw new Error("InstanceSet.fromJSON: runs do not cover count");
    } else if (obj.labels && obj.labels.length === obj.count) {
      for (let i = 0; i < obj.count; i++) {
        assertLabel(obj.labels[i]);
        set.labels[i] = obj.labels[i];
      }
    } else {
      throw new Error("InstanceSet.fromJSON: expected runs or labels");
    }
    return set;
  }
}
