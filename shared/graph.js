/**
 * Superpoint graph over a gaussian cloud (plan §3.2.B, phase F2). Pure JS:
 * no WebGPU, no dependencies, runs in Node, a Worker or the page.
 *
 * Pipeline (all O(N·k) after an O(N) spatial hash):
 *   1. kNN (k = 10) on gaussian means with a hash grid (cell = median of the
 *      largest scale, widened to keep a few points per cell).
 *   2. Edge weight  w_ij = exp(-d_M² / 2) · exp(-‖c_i − c_j‖² / σ_c²)
 *      with d_M the symmetric Mahalanobis distance from both covariances and
 *      c the SH0 colour (rgb in 0..1).
 *   3. Cut edges with w < threshold, connected components → superpoints,
 *      numbered by size (0 = largest).
 *   4. Optional weighted-majority label diffusion on the same graph.
 *
 * Gaussian layout (shared/splat-io.js): Float32Array N*12 =
 *   [x, y, z, opacity, sx, sy, sz, pad, qw, qx, qy, qz]  (linear scales).
 *
 * The idea follows THGS (superpoint graphs) and LUDVIG (graph diffusion);
 * this is an independent MIT reimplementation, no code is vendored.
 */

export const GAUSSIAN_FLOATS = 12;
export const SH_C0 = 0.28209479177387814;

export const DEFAULT_GRAPH_OPTIONS = Object.freeze({
  /** neighbours per gaussian */
  k: 10,
  /** grid cell size; 0 = automatic (see autoCellSize) */
  cellSize: 0,
  /** multiplier on the density-based cell estimate (points per cell ≈ factor³) */
  cellFactor: 1.5,
  /** max hash-grid rings searched around a point (bounds the kNN cost) */
  maxRings: 3,
  /** colour term sigma (rgb units, 0..1) */
  sigmaColor: 0.25,
  /** edges with w < threshold are cut before connected components */
  threshold: 0.3,
  /** smallest scale used to invert a covariance (avoids infinite d_M) */
  minScale: 1e-6,
});

const GOLDEN_RATIO_CONJUGATE = 0.618033988749895;

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

function assertGaussians(gaussians) {
  if (!(gaussians instanceof Float32Array) || gaussians.length % GAUSSIAN_FLOATS !== 0) {
    throw new Error(`gaussians must be a Float32Array with ${GAUSSIAN_FLOATS} floats per gaussian`);
  }
  return gaussians.length / GAUSSIAN_FLOATS;
}

function assertColors(colors, count) {
  if (colors == null) return null;
  if (!(colors instanceof Float32Array) || colors.length !== count * 3) {
    throw new Error(`colors must be a Float32Array of ${count * 3} floats (rgb per gaussian)`);
  }
  return colors;
}

function assertCsr(csr, count) {
  if (!csr || !(csr.offsets instanceof Uint32Array) || !(csr.neighbors instanceof Uint32Array)) {
    throw new Error("csr must have Uint32Array offsets and neighbors");
  }
  if (csr.offsets.length !== count + 1) {
    throw new Error(`csr.offsets length ${csr.offsets.length} != count + 1 (${count + 1})`);
  }
  if (csr.offsets[count] !== csr.neighbors.length) {
    throw new Error("csr.offsets[count] must equal csr.neighbors.length");
  }
}

function positiveNumber(name, value, fallback) {
  if (value == null) return fallback;
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive number, got ${value}`);
  return value;
}

/** Deterministic palette for group ids (0 = sin grupo, grey). Mirrors group_color() in gpu-renderer.js. */
export function groupColor(group) {
  if (!Number.isInteger(group) || group < 0) throw new Error(`group must be a non-negative integer, got ${group}`);
  if (group === 0) return [0.35, 0.35, 0.35];
  const h = (Math.fround(group * GOLDEN_RATIO_CONJUGATE) % 1 + 1) % 1;
  const s = 0.65;
  const v = 0.95;
  const hh = h * 6;
  const i = Math.floor(hh);
  const f = hh - i;
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

/** SH DC coefficients (first 3 of each 48-float record) → rgb in 0..1. */
export function shDcToRgb(sh, count) {
  if (!(sh instanceof Float32Array) || sh.length < count * 48) {
    throw new Error(`sh must have 48 floats per gaussian (${count})`);
  }
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const s = i * 48;
    for (let c = 0; c < 3; c++) {
      const v = SH_C0 * sh[s + c] + 0.5;
      out[i * 3 + c] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  }
  return out;
}

/** Median of max(sx, sy, sz) over (a stride sample of) the cloud; 0 for an empty cloud. */
export function medianLargestScale(gaussians, maxSamples = 262144) {
  const n = assertGaussians(gaussians);
  if (n === 0) return 0;
  const stride = Math.max(1, Math.floor(n / maxSamples));
  const m = Math.floor((n - 1) / stride) + 1;
  const s = new Float32Array(m);
  for (let i = 0, k = 0; i < n; i += stride, k++) {
    const o = i * GAUSSIAN_FLOATS;
    const v = Math.max(gaussians[o + 4], gaussians[o + 5], gaussians[o + 6]);
    s[k] = Number.isFinite(v) ? v : 0;
  }
  s.sort();
  return s[m >> 1];
}

/** Axis-aligned bounds of the gaussian means. */
export function cloudBounds(gaussians) {
  const n = assertGaussians(gaussians);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const o = i * GAUSSIAN_FLOATS;
    for (let a = 0; a < 3; a++) {
      const v = gaussians[o + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  return n ? { min, max } : { min: [0, 0, 0], max: [0, 0, 0] };
}

/**
 * Cell size for the hash grid: the plan's "median of the largest scale",
 * widened to the density estimate cbrt(volume / N) · cellFactor so that a
 * ring-1 search (27 cells) usually holds k neighbours.
 */
export function autoCellSize(gaussians, { cellFactor = DEFAULT_GRAPH_OPTIONS.cellFactor } = {}) {
  const n = assertGaussians(gaussians);
  const medianScale = medianLargestScale(gaussians);
  const bounds = cloudBounds(gaussians);
  const ext = [0, 1, 2].map((a) => Math.max(bounds.max[a] - bounds.min[a], 0));
  const volume = ext[0] * ext[1] * ext[2];
  const densityCell = n > 0 && volume > 0 ? Math.cbrt(volume / n) * cellFactor : 0;
  let cellSize = Math.max(medianScale, densityCell);
  if (!(cellSize > 0)) cellSize = Math.max(...ext, 1e-3) / 8;
  return { cellSize, medianScale, densityCell, bounds };
}

function nextPow2(x) {
  let p = 1;
  while (p < x) p *= 2;
  return p;
}

// Spatial hash (Teschner et al.) on integer cell coordinates.
function hashCell(ix, iy, iz, mask) {
  return (((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) >>> 0) & mask;
}

/**
 * Spatial layout shared by kNN and edge weights: slots sorted by cell so that
 * neighbouring gaussians are neighbours in memory. `order[slot]` is the
 * gaussian index, `slotOf[index]` its slot, and `cellRange(gx, gy, gz)` the
 * slot range of a cell (open-addressed hash on the linear cell index).
 */
function buildGridLayout(gaussians, n, cell, min) {
  const inv = 1 / cell;
  const cellOf = new Int32Array(n * 3);
  let dimX = 1, dimY = 1, dimZ = 1;
  for (let i = 0; i < n; i++) {
    const o = i * GAUSSIAN_FLOATS;
    const ix = Math.floor((gaussians[o] - min[0]) * inv);
    const iy = Math.floor((gaussians[o + 1] - min[1]) * inv);
    const iz = Math.floor((gaussians[o + 2] - min[2]) * inv);
    cellOf[i * 3] = ix;
    cellOf[i * 3 + 1] = iy;
    cellOf[i * 3 + 2] = iz;
    if (ix >= dimX) dimX = ix + 1;
    if (iy >= dimY) dimY = iy + 1;
    if (iz >= dimZ) dimZ = iz + 1;
  }
  // linear cell index (exact in a double while dims product < 2^31 and n < 2^22)
  const linearOk = dimX * dimY * dimZ < 2 ** 31 && n < 2 ** 22;
  const keys = new Float64Array(n);
  const tableSize = nextPow2(Math.max(16, n * 2));
  const mask = tableSize - 1;
  const linearOfCell = (ix, iy, iz) => (ix * dimY + iy) * dimZ + iz;
  for (let i = 0; i < n; i++) {
    const ix = cellOf[i * 3], iy = cellOf[i * 3 + 1], iz = cellOf[i * 3 + 2];
    const L = linearOk ? linearOfCell(ix, iy, iz) : hashCell(ix, iy, iz, 0x7fffffff);
    keys[i] = L * 4194304 + i; // 2^22
  }
  keys.sort();
  const order = new Uint32Array(n);
  const slotOf = new Uint32Array(n);
  for (let q = 0; q < n; q++) {
    const i = keys[q] % 4194304;
    order[q] = i;
    slotOf[i] = q;
  }
  const spos = new Float32Array(n * 3);
  const scell = new Int32Array(n * 3);
  for (let q = 0; q < n; q++) {
    const i = order[q];
    const o = i * GAUSSIAN_FLOATS;
    spos[q * 3] = gaussians[o];
    spos[q * 3 + 1] = gaussians[o + 1];
    spos[q * 3 + 2] = gaussians[o + 2];
    scell[q * 3] = cellOf[i * 3];
    scell[q * 3 + 1] = cellOf[i * 3 + 1];
    scell[q * 3 + 2] = cellOf[i * 3 + 2];
  }
  // open-addressed table: cell key → [start, end) slot run
  const tKey = new Float64Array(tableSize).fill(-1);
  const tStart = new Uint32Array(tableSize);
  const tEnd = new Uint32Array(tableSize);
  let q = 0;
  let cells = 0;
  while (q < n) {
    const ix = scell[q * 3], iy = scell[q * 3 + 1], iz = scell[q * 3 + 2];
    let e = q + 1;
    while (e < n && scell[e * 3] === ix && scell[e * 3 + 1] === iy && scell[e * 3 + 2] === iz) e++;
    const L = linearOk ? linearOfCell(ix, iy, iz) : hashCell(ix, iy, iz, 0x7fffffff);
    let h = hashCell(ix, iy, iz, mask);
    while (tKey[h] !== -1) h = (h + 1) & mask;
    tKey[h] = L;
    tStart[h] = q;
    tEnd[h] = e;
    cells++;
    q = e;
  }
  // returns start slot, end via rangeEnd (avoids allocating per lookup)
  let lastEnd = 0;
  const cellRange = (ix, iy, iz) => {
    const L = linearOk ? linearOfCell(ix, iy, iz) : hashCell(ix, iy, iz, 0x7fffffff);
    let h = hashCell(ix, iy, iz, mask);
    for (;;) {
      const key = tKey[h];
      if (key === L) {
        // hash-mode keys can collide for different cells; the caller re-checks scell
        lastEnd = tEnd[h];
        return tStart[h];
      }
      if (key === -1) {
        lastEnd = 0;
        return 0;
      }
      h = (h + 1) & mask;
    }
  };
  const rangeEnd = () => lastEnd;
  return { order, slotOf, spos, scell, cellRange, rangeEnd, cells, dims: [dimX, dimY, dimZ], linearOk };
}

/**
 * kNN graph on gaussian means, symmetrised (undirected, no duplicates) and
 * stored as CSR over ORIGINAL gaussian indices. Neighbours beyond `maxRings`
 * cells are never found, so isolated points may end up with fewer than k
 * neighbours (or none). The returned `layout` lets edgeWeights() work in the
 * cache-friendly slot order.
 *
 * @returns {{count:number, k:number, cellSize:number, offsets:Uint32Array,
 *   neighbors:Uint32Array, dist2:Float32Array, stats:object, layout:object}}
 */
export function buildKnnGraph(gaussians, options = {}) {
  const n = assertGaussians(gaussians);
  const t0 = now();
  const kReq = options.k == null ? DEFAULT_GRAPH_OPTIONS.k : options.k;
  if (!Number.isInteger(kReq) || kReq < 1) throw new Error(`k must be a positive integer, got ${kReq}`);
  const maxRings = options.maxRings == null ? DEFAULT_GRAPH_OPTIONS.maxRings : options.maxRings;
  if (!Number.isInteger(maxRings) || maxRings < 1) throw new Error(`maxRings must be >= 1, got ${maxRings}`);
  const k = Math.min(kReq, Math.max(0, n - 1));
  if (n === 0 || k === 0) {
    return {
      count: n,
      k,
      cellSize: 0,
      offsets: new Uint32Array(n + 1),
      neighbors: new Uint32Array(0),
      dist2: new Float32Array(0),
      stats: { msGrid: 0, msKnn: 0, msSymmetrize: 0, avgDegree: 0, truncated: 0 },
      layout: null,
    };
  }

  // ---- grid in slot order (see buildGridLayout)
  const auto = autoCellSize(gaussians, options);
  const cell = positiveNumber("cellSize", options.cellSize || null, auto.cellSize);
  const layout = buildGridLayout(gaussians, n, cell, auto.bounds.min);
  const { order, spos, scell, cellRange, rangeEnd } = layout;
  const inv = 1 / cell;
  const [minX, minY, minZ] = auto.bounds.min;
  const t1 = now();

  // ---- directed kNN in slot space
  const knnSlot = new Int32Array(n * k).fill(-1);
  const knnD2 = new Float32Array(n * k).fill(Infinity);
  let truncated = 0;
  for (let q = 0; q < n; q++) {
    const px = spos[q * 3];
    const py = spos[q * 3 + 1];
    const pz = spos[q * 3 + 2];
    const ix = scell[q * 3];
    const iy = scell[q * 3 + 1];
    const iz = scell[q * 3 + 2];
    const base = q * k;
    let found = 0;
    let worst = Infinity; // knnD2[base + k - 1]
    // distance from the point to the nearest face of its own cell: after ring r
    // every point closer than r·cell + dEdge lies inside the searched block
    const fx = (px - minX) * inv - ix;
    const fy = (py - minY) * inv - iy;
    const fz = (pz - minZ) * inv - iz;
    const dEdge = cell * Math.min(fx, 1 - fx, fy, 1 - fy, fz, 1 - fz);
    for (let r = 0; r <= maxRings; r++) {
      for (let dx = -r; dx <= r; dx++) {
        const ex = dx === r || dx === -r;
        for (let dy = -r; dy <= r; dy++) {
          const inner = !(ex || dy === r || dy === -r); // then only dz = ±r is on the shell
          for (let dz = -r; dz <= r; dz++) {
            if (inner && dz !== r && dz !== -r) continue;
            const gx = ix + dx;
            const gy = iy + dy;
            const gz = iz + dz;
            const start = cellRange(gx, gy, gz);
            const end = rangeEnd();
            for (let s = start; s < end; s++) {
              if (s === q || scell[s * 3] !== gx || scell[s * 3 + 1] !== gy || scell[s * 3 + 2] !== gz) continue;
              const ddx = spos[s * 3] - px;
              const ddy = spos[s * 3 + 1] - py;
              const ddz = spos[s * 3 + 2] - pz;
              const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
              if (d2 >= worst) continue;
              let pos = found < k ? found : k - 1; // insertion into the sorted top-k
              while (pos > 0 && knnD2[base + pos - 1] > d2) {
                knnD2[base + pos] = knnD2[base + pos - 1];
                knnSlot[base + pos] = knnSlot[base + pos - 1];
                pos--;
              }
              knnD2[base + pos] = d2;
              knnSlot[base + pos] = s;
              if (found < k) found++;
              if (found === k) worst = knnD2[base + k - 1];
            }
          }
        }
      }
      const bound = r * cell + dEdge;
      if (found === k && worst <= bound * bound) break;
      if (r === maxRings && found < k) truncated++;
    }
  }
  const t2 = now();

  // ---- symmetrise in slot space: union of both directions, then per-node
  // sort + dedupe (lists are short, so this stays cache-local)
  const deg = new Uint32Array(n);
  for (let q = 0; q < n; q++) {
    const b = q * k;
    for (let t = 0; t < k; t++) {
      const s = knnSlot[b + t];
      if (s < 0) break;
      deg[q]++;
      deg[s]++;
    }
  }
  const offS = new Uint32Array(n + 1);
  for (let q = 0; q < n; q++) offS[q + 1] = offS[q] + deg[q];
  const fill = offS.slice(0, n);
  const nbrS = new Int32Array(offS[n]);
  const d2S = new Float32Array(offS[n]);
  for (let q = 0; q < n; q++) {
    const b = q * k;
    for (let t = 0; t < k; t++) {
      const s = knnSlot[b + t];
      if (s < 0) break;
      const d2 = knnD2[b + t];
      nbrS[fill[q]] = s;
      d2S[fill[q]++] = d2;
      nbrS[fill[s]] = q;
      d2S[fill[s]++] = d2;
    }
  }
  // drop duplicate neighbours (mutual pairs appear twice) with a stamp array;
  // lists stay in insertion order, which is fine for every consumer
  const offsetsS = new Uint32Array(n + 1);
  const stamp = new Uint32Array(n);
  let w = 0;
  for (let q = 0; q < n; q++) {
    const a = offS[q];
    const b = offS[q + 1];
    const tag = q + 1;
    for (let x = a; x < b; x++) {
      const v = nbrS[x];
      if (stamp[v] === tag) continue;
      stamp[v] = tag;
      nbrS[w] = v;
      d2S[w++] = d2S[x];
    }
    offsetsS[q + 1] = w;
  }
  // ---- map back to original indices
  const offsets = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) {
    const q = layout.slotOf[i];
    offsets[i + 1] = offsets[i] + (offsetsS[q + 1] - offsetsS[q]);
  }
  const neighbors = new Uint32Array(w);
  const dist2 = new Float32Array(w);
  for (let i = 0; i < n; i++) {
    const q = layout.slotOf[i];
    let e = offsets[i];
    for (let x = offsetsS[q]; x < offsetsS[q + 1]; x++) {
      neighbors[e] = order[nbrS[x]];
      dist2[e++] = d2S[x];
    }
  }
  const t3 = now();
  layout.slotOffsets = offsetsS;
  layout.slotNeighbors = nbrS.subarray(0, w);
  return {
    count: n,
    k,
    cellSize: cell,
    offsets,
    neighbors,
    dist2,
    layout,
    stats: {
      msGrid: t1 - t0,
      msKnn: t2 - t1,
      msSymmetrize: t3 - t2,
      avgDegree: w / n,
      truncated,
      cells: layout.cells,
      medianScale: auto.medianScale,
      densityCell: auto.densityCell,
    },
  };
}

/** Inverse covariances Σ⁻¹ = R·diag(1/s²)·Rᵀ, packed [a, b, c, d, e, f] for [[a,b,c],[b,d,e],[c,e,f]]. */
export function inverseCovariances(gaussians, minScale = DEFAULT_GRAPH_OPTIONS.minScale) {
  const n = assertGaussians(gaussians);
  const out = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {
    const o = i * GAUSSIAN_FLOATS;
    let sx = gaussians[o + 4];
    let sy = gaussians[o + 5];
    let sz = gaussians[o + 6];
    if (!(sx > minScale)) sx = minScale;
    if (!(sy > minScale)) sy = minScale;
    if (!(sz > minScale)) sz = minScale;
    let w = gaussians[o + 8];
    let x = gaussians[o + 9];
    let y = gaussians[o + 10];
    let z = gaussians[o + 11];
    let qn = Math.hypot(w, x, y, z);
    if (!(qn > 0)) { w = 1; x = y = z = 0; qn = 1; }
    w /= qn; x /= qn; y /= qn; z /= qn;
    // rotation matrix columns (same convention as gpu-renderer.js cov3d_from_quat)
    const r00 = 1 - 2 * (y * y + z * z), r10 = 2 * (x * y + w * z), r20 = 2 * (x * z - w * y);
    const r01 = 2 * (x * y - w * z), r11 = 1 - 2 * (x * x + z * z), r21 = 2 * (y * z + w * x);
    const r02 = 2 * (x * z + w * y), r12 = 2 * (y * z - w * x), r22 = 1 - 2 * (x * x + y * y);
    const ia = 1 / (sx * sx), ib = 1 / (sy * sy), ic = 1 / (sz * sz);
    // Σ⁻¹ = Σ_k inv_k · col_k · col_kᵀ
    const p = i * 6;
    out[p] = ia * r00 * r00 + ib * r01 * r01 + ic * r02 * r02;
    out[p + 1] = ia * r00 * r10 + ib * r01 * r11 + ic * r02 * r12;
    out[p + 2] = ia * r00 * r20 + ib * r01 * r21 + ic * r02 * r22;
    out[p + 3] = ia * r10 * r10 + ib * r11 * r11 + ic * r12 * r12;
    out[p + 4] = ia * r10 * r20 + ib * r11 * r21 + ic * r12 * r22;
    out[p + 5] = ia * r20 * r20 + ib * r21 * r21 + ic * r22 * r22;
  }
  return out;
}

/**
 * Edge weights for a CSR graph: w = exp(-d_M²/2) · exp(-‖Δc‖²/σ_c²).
 * d_M² = ½(Δᵀ Σ_i⁻¹ Δ + Δᵀ Σ_j⁻¹ Δ). Weights are symmetric; `colors` may be null
 * (colour term = 1).
 */
export function edgeWeights(gaussians, colors, csr, options = {}) {
  const n = assertGaussians(gaussians);
  assertCsr(csr, n);
  const col = assertColors(colors, n);
  const sigmaColor = positiveNumber("sigmaColor", options.sigmaColor, DEFAULT_GRAPH_OPTIONS.sigmaColor);
  const invSig2 = 1 / (sigmaColor * sigmaColor);
  const ic = inverseCovariances(gaussians, options.minScale);
  const { offsets, neighbors } = csr;
  const layout = csr.layout && csr.layout.slotOffsets ? csr.layout : null;
  if (!layout) {
    // generic path: original index order
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const o = i * GAUSSIAN_FLOATS;
      pos[i * 3] = gaussians[o];
      pos[i * 3 + 1] = gaussians[o + 1];
      pos[i * 3 + 2] = gaussians[o + 2];
    }
    return weightsOverCsr(offsets, neighbors, pos, ic, col, invSig2);
  }
  // Slot space (CSR from buildKnnGraph): permuted copies so a node and its
  // neighbours sit close in memory; edge t of node i in the original CSR is
  // edge t of slot slotOf[i], so the weights map back 1:1.
  const P = layout.order;
  const icS = new Float32Array(n * 6);
  const colS = col ? new Float32Array(n * 3) : null;
  for (let q = 0; q < n; q++) {
    const i = P[q];
    const a = i * 6;
    const b = q * 6;
    icS[b] = ic[a]; icS[b + 1] = ic[a + 1]; icS[b + 2] = ic[a + 2];
    icS[b + 3] = ic[a + 3]; icS[b + 4] = ic[a + 4]; icS[b + 5] = ic[a + 5];
    if (colS) {
      colS[q * 3] = col[i * 3];
      colS[q * 3 + 1] = col[i * 3 + 1];
      colS[q * 3 + 2] = col[i * 3 + 2];
    }
  }
  const wS = weightsOverCsr(layout.slotOffsets, layout.slotNeighbors, layout.spos, icS, colS, invSig2);
  const weights = new Float32Array(neighbors.length);
  const { slotOf, slotOffsets } = layout;
  for (let i = 0; i < n; i++) {
    const q = slotOf[i];
    let e = offsets[i];
    for (let x = slotOffsets[q]; x < slotOffsets[q + 1]; x++) weights[e++] = wS[x];
  }
  return weights;
}

/** w = exp(-d_M²/2 − ‖Δc‖²/σ²) over a CSR whose nodes index pos (N*3), ic (N*6) and col (N*3|null). */
function weightsOverCsr(offsets, neighbors, pos, ic, col, invSig2) {
  const n = offsets.length - 1;
  const weights = new Float32Array(neighbors.length);
  for (let a = 0; a < n; a++) {
    const ax = pos[a * 3];
    const ay = pos[a * 3 + 1];
    const az = pos[a * 3 + 2];
    const pa = a * 6;
    const ia = ic[pa], ib = ic[pa + 1], ic2 = ic[pa + 2], id = ic[pa + 3], ie = ic[pa + 4], iff = ic[pa + 5];
    for (let e = offsets[a]; e < offsets[a + 1]; e++) {
      const b = neighbors[e];
      const dx = pos[b * 3] - ax;
      const dy = pos[b * 3 + 1] - ay;
      const dz = pos[b * 3 + 2] - az;
      const pb = b * 6;
      const qa = ia * dx * dx + id * dy * dy + iff * dz * dz + 2 * (ib * dx * dy + ic2 * dx * dz + ie * dy * dz);
      const qb =
        ic[pb] * dx * dx + ic[pb + 3] * dy * dy + ic[pb + 5] * dz * dz +
        2 * (ic[pb + 1] * dx * dy + ic[pb + 2] * dx * dz + ic[pb + 4] * dy * dz);
      let arg = -0.25 * (qa + qb); // -d_M²/2 with d_M² = ½(qa + qb)
      if (col) {
        const cr = col[b * 3] - col[a * 3];
        const cg = col[b * 3 + 1] - col[a * 3 + 1];
        const cb = col[b * 3 + 2] - col[a * 3 + 2];
        arg -= (cr * cr + cg * cg + cb * cb) * invSig2;
      }
      const wgt = Math.exp(arg);
      weights[e] = wgt === wgt ? wgt : 0; // NaN guard
    }
  }
  return weights;
}

/**
 * Connected components of the edges with weight ≥ threshold (union-find),
 * renumbered by size (component 0 is the largest; ties by first index).
 * @returns {{component:Uint32Array, count:number, sizes:Uint32Array}}
 */
export function connectedComponents(csr, weights, threshold = DEFAULT_GRAPH_OPTIONS.threshold) {
  const n = csr.offsets.length - 1;
  assertCsr(csr, n);
  if (weights != null && weights.length !== csr.neighbors.length) {
    throw new Error("weights length must equal csr.neighbors.length");
  }
  if (!Number.isFinite(threshold)) throw new Error(`threshold must be a number, got ${threshold}`);
  const parent = new Uint32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (a) => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  const { offsets, neighbors } = csr;
  for (let i = 0; i < n; i++) {
    for (let e = offsets[i]; e < offsets[i + 1]; e++) {
      const j = neighbors[e];
      if (j < i) continue; // each undirected edge once
      if (weights && weights[e] < threshold) continue;
      const ri = find(i);
      const rj = find(j);
      if (ri !== rj) parent[Math.max(ri, rj)] = Math.min(ri, rj);
    }
  }
  const root = new Uint32Array(n);
  const rootSize = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const r = find(i);
    root[i] = r;
    rootSize[r]++;
  }
  // order roots by size desc, then by index
  const roots = [];
  for (let i = 0; i < n; i++) if (rootSize[i] > 0) roots.push(i);
  roots.sort((a, b) => rootSize[b] - rootSize[a] || a - b);
  const idOfRoot = new Uint32Array(n);
  const sizes = new Uint32Array(roots.length);
  for (let c = 0; c < roots.length; c++) {
    idOfRoot[roots[c]] = c;
    sizes[c] = rootSize[roots[c]];
  }
  const component = new Uint32Array(n);
  for (let i = 0; i < n; i++) component[i] = idOfRoot[root[i]];
  return { component, count: roots.length, sizes };
}

/** Mean position per component. @returns {Float32Array} count*3 */
export function componentCentroids(component, count, gaussians) {
  const n = assertGaussians(gaussians);
  if (component.length !== n) throw new Error("component length must equal the gaussian count");
  const sum = new Float64Array(count * 3);
  const num = new Uint32Array(count);
  for (let i = 0; i < n; i++) {
    const c = component[i];
    if (c >= count) throw new Error(`component id ${c} >= count ${count}`);
    const o = i * GAUSSIAN_FLOATS;
    sum[c * 3] += gaussians[o];
    sum[c * 3 + 1] += gaussians[o + 1];
    sum[c * 3 + 2] += gaussians[o + 2];
    num[c]++;
  }
  const out = new Float32Array(count * 3);
  for (let c = 0; c < count; c++) {
    const d = num[c] || 1;
    out[c * 3] = sum[c * 3] / d;
    out[c * 3 + 1] = sum[c * 3 + 1] / d;
    out[c * 3 + 2] = sum[c * 3 + 2] / d;
  }
  return out;
}

/**
 * Full F2 pipeline: kNN → weights → threshold → components (superpoints).
 * @param {Float32Array} gaussians N*12
 * @param {Float32Array|null} colors N*3 rgb in 0..1 (see shDcToRgb), or null
 * @param {object} options see DEFAULT_GRAPH_OPTIONS
 * @returns {{count:number, k:number, cellSize:number, threshold:number,
 *   csr:{offsets:Uint32Array, neighbors:Uint32Array, dist2:Float32Array, weights:Float32Array},
 *   superpoint:Uint32Array, superpointCount:number, sizes:Uint32Array, centroids:Float32Array,
 *   stats:object}}
 */
export function buildSuperpointGraph(gaussians, colors = null, options = {}) {
  const n = assertGaussians(gaussians);
  assertColors(colors, n);
  const opts = { ...DEFAULT_GRAPH_OPTIONS, ...options };
  const t0 = now();
  const knn = buildKnnGraph(gaussians, opts);
  const csr = { offsets: knn.offsets, neighbors: knn.neighbors, dist2: knn.dist2, layout: knn.layout };
  const t1 = now();
  const weights = edgeWeights(gaussians, colors, csr, opts);
  const t2 = now();
  const cc = connectedComponents(csr, weights, opts.threshold);
  const t3 = now();
  const centroids = componentCentroids(cc.component, cc.count, gaussians);
  const t4 = now();
  let edgesKept = 0;
  for (let e = 0; e < weights.length; e++) if (weights[e] >= opts.threshold) edgesKept++;
  return {
    count: n,
    k: knn.k,
    cellSize: knn.cellSize,
    threshold: opts.threshold,
    sigmaColor: opts.sigmaColor,
    csr: { offsets: csr.offsets, neighbors: csr.neighbors, dist2: csr.dist2, weights },
    superpoint: cc.component,
    superpointCount: cc.count,
    sizes: cc.sizes,
    centroids,
    stats: {
      ...knn.stats,
      msWeights: t2 - t1,
      msComponents: t3 - t2,
      msCentroids: t4 - t3,
      msTotal: t4 - t0,
      edges: weights.length / 2,
      edgesKept: edgesKept / 2,
    },
  };
}

/**
 * Weighted-majority label diffusion (LUDVIG-style cleanup). Each iteration
 * every non-seed gaussian takes the label with the largest summed edge
 * weight among its neighbours, its own label counting selfWeight · Σw.
 * Label 0 (fondo) participates like any other label.
 *
 * @param {Uint32Array} labels
 * @param {{offsets:Uint32Array, neighbors:Uint32Array}} csr
 * @param {Float32Array|null} weights per edge (null = 1)
 * @param {{iterations?:number, seeds?:Uint8Array|null, selfWeight?:number}} options
 * @returns {Uint32Array} new labels (input untouched)
 */
export function diffuseLabels(labels, csr, weights = null, options = {}) {
  const n = labels.length;
  assertCsr(csr, n);
  if (weights != null && weights.length !== csr.neighbors.length) {
    throw new Error("weights length must equal csr.neighbors.length");
  }
  const iterations = options.iterations == null ? 5 : options.iterations;
  if (!Number.isInteger(iterations) || iterations < 0) throw new Error(`iterations must be >= 0, got ${iterations}`);
  const seeds = options.seeds || null;
  if (seeds && seeds.length !== n) throw new Error("seeds length must equal labels length");
  const selfWeight = options.selfWeight == null ? 0.5 : options.selfWeight;
  const { offsets, neighbors } = csr;
  let cur = Uint32Array.from(labels);
  let next = new Uint32Array(n);
  // scratch tallies: at most deg+1 distinct labels per node
  let maxDeg = 0;
  for (let i = 0; i < n; i++) maxDeg = Math.max(maxDeg, offsets[i + 1] - offsets[i]);
  const tl = new Uint32Array(maxDeg + 1);
  const tw = new Float64Array(maxDeg + 1);
  for (let it = 0; it < iterations; it++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const own = cur[i];
      if (seeds && seeds[i]) { next[i] = own; continue; }
      let m = 0;
      let total = 0;
      for (let e = offsets[i]; e < offsets[i + 1]; e++) {
        const w = weights ? weights[e] : 1;
        if (!(w > 0)) continue;
        const l = cur[neighbors[e]];
        total += w;
        let t = 0;
        while (t < m && tl[t] !== l) t++;
        if (t === m) { tl[m] = l; tw[m] = w; m++; } else tw[t] += w;
      }
      let best = own;
      let bestW = selfWeight * total;
      for (let t = 0; t < m; t++) {
        // strict > keeps the current label on ties
        if (tw[t] > bestW || (tw[t] === bestW && tl[t] === own)) { bestW = tw[t]; best = tl[t]; }
      }
      next[i] = best;
      if (best !== own) changed++;
    }
    const tmp = cur; cur = next; next = tmp;
    if (changed === 0) break;
  }
  return cur;
}

/**
 * Map superpoints (already ordered by size) to F1 instance labels 1..maxLabels;
 * smaller / overflowing groups become 0 (fondo).
 * @returns {{labels:Uint32Array, groupOfLabel:Uint32Array}} groupOfLabel[label] = superpoint id
 */
export function groupsToLabels(superpoint, sizes, { maxLabels = 4095, minSize = 1 } = {}) {
  const labels = new Uint32Array(superpoint.length);
  const used = Math.min(sizes.length, maxLabels);
  let last = 0;
  for (let g = 0; g < used; g++) if (sizes[g] >= minSize) last = g + 1;
  const groupOfLabel = new Uint32Array(last + 1);
  for (let g = 0; g < last; g++) groupOfLabel[g + 1] = g;
  for (let i = 0; i < superpoint.length; i++) {
    const g = superpoint[i];
    labels[i] = g < last && sizes[g] >= minSize ? g + 1 : 0;
  }
  return { labels, groupOfLabel };
}

/** Indices of the gaussians in superpoint `group` (ascending). */
export function indicesOfGroup(superpoint, group) {
  let n = 0;
  for (let i = 0; i < superpoint.length; i++) if (superpoint[i] === group) n++;
  const out = new Uint32Array(n);
  let k = 0;
  for (let i = 0; i < superpoint.length; i++) if (superpoint[i] === group) out[k++] = i;
  return out;
}
