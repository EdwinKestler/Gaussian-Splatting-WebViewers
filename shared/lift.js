/**
 * F3 mask lifting (plan §3.2.C): from per-view contribution matrices
 * (gpu-renderer renderContributions → C_v[i][l] = Σ α·T over pixels of mask
 * label l) to one instance label per gaussian.
 *
 *   1. Per view: closed-form FlashSplat assignment
 *        label_i = argmax_l C[i][l]  with a background bias on column 0.
 *   2. Cross-view association (Gaga-style memory, without training): every
 *      (view, mask) is described by the mass-weighted superpoint histogram of
 *      the gaussians it captured; a mask joins the instance whose accumulated
 *      histogram contains it (Σ min over the smaller total ≥ iouThreshold),
 *      otherwise it starts a new instance. Superpoints (F2) matter: two
 *      cameras 90° apart lift different gaussians of the same object, and
 *      only the shared superpoints reveal that they are one instance.
 *   3. Global assignment on the summed contributions of each global instance.
 *   4. Optional cleanup with shared/graph.js diffuseLabels (done by the caller).
 *
 * Pure JS, no dependencies; importable from Node for tests.
 */

export const DEFAULT_LIFT_OPTIONS = Object.freeze({
  /** score_0 = (1 + backgroundBias) · C[i][0]; higher = more conservative foreground */
  backgroundBias: 0.3,
  /** gaussians whose best foreground mass is below this stay fondo (0) */
  minMass: 1e-3,
  /** overlap (containment of the smaller superpoint histogram) needed to merge a mask into an instance */
  iouThreshold: 0.5,
  /** ignore masks that lifted fewer gaussians than this when associating */
  minGaussians: 5,
});

function assertContrib(contrib, count, labelCount) {
  if (!(contrib instanceof Float32Array) || contrib.length !== count * labelCount) {
    throw new Error(`contrib must be a Float32Array of count·labelCount = ${count * labelCount} floats`);
  }
}

/**
 * FlashSplat closed-form assignment for one contribution matrix.
 * @param {Float32Array} contrib count × labelCount (column 0 = fondo)
 * @param {number} count
 * @param {number} labelCount
 * @param {{backgroundBias?:number, minMass?:number}} [options]
 * @returns {Uint32Array} label per gaussian (0 = fondo / unseen)
 */
export function assignLabels(contrib, count, labelCount, options = {}) {
  assertContrib(contrib, count, labelCount);
  const bias = options.backgroundBias == null ? DEFAULT_LIFT_OPTIONS.backgroundBias : options.backgroundBias;
  const minMass = options.minMass == null ? DEFAULT_LIFT_OPTIONS.minMass : options.minMass;
  if (!Number.isFinite(bias) || bias < -1) throw new Error(`backgroundBias must be a number >= -1, got ${bias}`);
  const labels = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const row = i * labelCount;
    const bg = (1 + bias) * contrib[row];
    let best = 0;
    let bestMass = 0;
    for (let l = 1; l < labelCount; l++) {
      const m = contrib[row + l];
      if (m > bestMass) {
        bestMass = m;
        best = l;
      }
    }
    labels[i] = bestMass >= minMass && bestMass > bg ? best : 0;
  }
  return labels;
}

/**
 * Sparse histogram (Map key → weight) of the gaussians carrying `label`,
 * keyed by superpoint id (or by gaussian index when superpoint is null).
 */
export function maskHistogram(labels, label, superpoint = null, weights = null) {
  const hist = new Map();
  let total = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== label) continue;
    const key = superpoint ? superpoint[i] : i;
    const w = weights ? weights[i] : 1;
    hist.set(key, (hist.get(key) || 0) + w);
    total += w;
  }
  return { hist, total };
}

/** Weighted Jaccard overlap of two sparse histograms: Σ min / Σ max. */
export function weightedJaccard(a, b) {
  let inter = 0;
  let union = 0;
  for (const [k, va] of a) {
    const vb = b.get(k) || 0;
    inter += Math.min(va, vb);
    union += Math.max(va, vb);
  }
  for (const [k, vb] of b) if (!a.has(k)) union += vb;
  return union > 0 ? inter / union : 0;
}

/** Overlap of two sparse histograms relative to the smaller one: Σ min / min(Σa, Σb). */
export function containment(a, totalA, b, totalB) {
  let inter = 0;
  for (const [k, va] of a) {
    const vb = b.get(k);
    if (vb) inter += Math.min(va, vb);
  }
  const denom = Math.min(totalA, totalB);
  return denom > 0 ? inter / denom : 0;
}

/**
 * Associate view-local masks into global instances (Gaga-style 3D memory
 * bank, no training). Views are processed in order; every mask is described
 * by the histogram of the superpoints (or gaussians) it lifted, weighted by
 * the mask's contribution mass when `masses` is given. A mask joins the
 * existing instance whose accumulated histogram it overlaps most, when that
 * overlap (Σ min over the smaller total, i.e. containment) reaches
 * iouThreshold; otherwise it starts a new instance. Containment rather than
 * Jaccard lets a partial view of an object (one side of a chair) join the
 * instance built from the other views.
 *
 * @param {Array<{labels:Uint32Array, labelCount:number, masses?:Float32Array|null}>} views
 *   per-view assignments (assignLabels output); masses[i] = lifted mass of gaussian i (optional)
 * @param {{superpoint?:Uint32Array|null, iouThreshold?:number, minGaussians?:number, mode?:"containment"|"jaccard"}} [options]
 * @returns {{globalCount:number, globalOf:Uint32Array[], members:Array<Array<[number, number]>>,
 *   pairs:Array<{a:[number,number], b:number, overlap:number}>}}
 *   globalOf[v][localLabel] = global id (0 = fondo / dropped); members[g-1] = [[view, local], ...];
 *   pairs lists every merge decision (mask a joined instance b with the given overlap)
 */
export function associateMasks(views, options = {}) {
  const superpoint = options.superpoint || null;
  const threshold = options.iouThreshold == null ? DEFAULT_LIFT_OPTIONS.iouThreshold : options.iouThreshold;
  const minGaussians = options.minGaussians == null ? DEFAULT_LIFT_OPTIONS.minGaussians : options.minGaussians;
  const mode = options.mode || "containment";
  const instances = []; // { hist: Map, total, members: [[view, local]] }
  const globalOf = views.map((v) => new Uint32Array(v.labelCount));
  const pairs = [];
  views.forEach((view, vi) => {
    if (superpoint && superpoint.length !== view.labels.length) {
      throw new Error("superpoint length must equal the label count");
    }
    for (let l = 1; l < view.labelCount; l++) {
      const { hist, total } = maskHistogram(view.labels, l, superpoint, view.masses || null);
      let count = 0;
      for (let i = 0; i < view.labels.length; i++) if (view.labels[i] === l) count++;
      if (count < minGaussians || total <= 0) continue;
      let best = -1;
      let bestOverlap = 0;
      for (let g = 0; g < instances.length; g++) {
        const inst = instances[g];
        const ov = mode === "jaccard" ? weightedJaccard(hist, inst.hist) : containment(hist, total, inst.hist, inst.total);
        if (ov > bestOverlap) {
          bestOverlap = ov;
          best = g;
        }
      }
      if (best >= 0 && bestOverlap >= threshold) {
        const inst = instances[best];
        for (const [k, w] of hist) inst.hist.set(k, (inst.hist.get(k) || 0) + w);
        inst.total += total;
        inst.members.push([vi, l]);
        globalOf[vi][l] = best + 1;
        pairs.push({ a: [vi, l], b: best + 1, overlap: bestOverlap });
      } else {
        instances.push({ hist: new Map(hist), total, members: [[vi, l]] });
        globalOf[vi][l] = instances.length;
      }
    }
  });
  // renumber by accumulated mass (largest first), stable by creation order
  const order = instances.map((inst, g) => g).sort((x, y) => instances[y].total - instances[x].total || x - y);
  const newId = new Uint32Array(instances.length);
  order.forEach((g, k) => (newId[g] = k + 1));
  for (const map of globalOf) for (let l = 0; l < map.length; l++) if (map[l]) map[l] = newId[map[l] - 1];
  for (const pr of pairs) pr.b = newId[pr.b - 1];
  const members = order.map((g) => instances[g].members);
  return { globalCount: instances.length, globalOf, members, pairs };
}

/**
 * Sum per-view contribution matrices into a global count × (globalCount + 1) matrix.
 * Column 0 (fondo) is summed as well; local labels mapped to 0 are dropped.
 */
export function mergeContributions(views, globalOf, count, globalCount) {
  const cols = globalCount + 1;
  const out = new Float32Array(count * cols);
  views.forEach((view, vi) => {
    assertContrib(view.contrib, count, view.labelCount);
    const map = globalOf[vi];
    const lc = view.labelCount;
    for (let i = 0; i < count; i++) {
      const src = i * lc;
      const dst = i * cols;
      out[dst] += view.contrib[src];
      for (let l = 1; l < lc; l++) {
        const g = map[l];
        if (g) out[dst + g] += view.contrib[src + l];
      }
    }
  });
  return out;
}

/**
 * Full lift: per-view assignment → association → global assignment.
 * @param {Array<{contrib:Float32Array, labelCount:number, names?:string[]}>} views
 * @param {{count:number, superpoint?:Uint32Array|null, backgroundBias?:number, minMass?:number,
 *   iouThreshold?:number, minGaussians?:number}} options
 * @returns {{labels:Uint32Array, globalCount:number, contrib:Float32Array, association:object,
 *   perView:Uint32Array[], names:string[]}}
 */
export function liftViews(views, options) {
  if (!Array.isArray(views) || views.length === 0) throw new Error("liftViews needs at least one view");
  const count = options.count;
  if (!Number.isInteger(count) || count < 0) throw new Error(`count must be a non-negative integer, got ${count}`);
  const perView = views.map((v) => assignLabels(v.contrib, count, v.labelCount, options));
  const masses = views.map((v, vi) => {
    const m = new Float32Array(count);
    const lc = v.labelCount;
    for (let i = 0; i < count; i++) {
      const l = perView[vi][i];
      if (l) m[i] = v.contrib[i * lc + l];
    }
    return m;
  });
  const association = associateMasks(
    views.map((v, i) => ({ labels: perView[i], labelCount: v.labelCount, masses: masses[i] })),
    options
  );
  const contrib = mergeContributions(views, association.globalOf, count, association.globalCount);
  const labels = assignLabels(contrib, count, association.globalCount + 1, options);
  // names: first non-empty local name of each global instance
  const names = new Array(association.globalCount + 1).fill("");
  association.members.forEach((list, k) => {
    for (const [vi, local] of list) {
      const n = views[vi].names && views[vi].names[local];
      if (n) {
        names[k + 1] = n;
        break;
      }
    }
    if (!names[k + 1]) names[k + 1] = `objeto ${k + 1}`;
  });
  return { labels, globalCount: association.globalCount, contrib, association, perView, names };
}

/** 3D IoU of one label between two labelings (for tests and reports). */
export function labelIou(a, b, labelA, labelB = labelA) {
  if (a.length !== b.length) throw new Error("labelings must have the same length");
  let inter = 0;
  let union = 0;
  for (let i = 0; i < a.length; i++) {
    const ia = a[i] === labelA;
    const ib = b[i] === labelB;
    if (ia && ib) inter++;
    if (ia || ib) union++;
  }
  return union ? inter / union : 1;
}

/**
 * Best-matching IoU per ground-truth label (labels may be renumbered).
 * @returns {Map<number, {label:number, iou:number}>} truthLabel → best predicted label
 */
export function matchLabels(truth, predicted) {
  const truthLabels = new Set();
  const predLabels = new Set();
  for (let i = 0; i < truth.length; i++) {
    if (truth[i]) truthLabels.add(truth[i]);
    if (predicted[i]) predLabels.add(predicted[i]);
  }
  const out = new Map();
  for (const t of truthLabels) {
    let best = { label: 0, iou: 0 };
    for (const p of predLabels) {
      const iou = labelIou(truth, predicted, t, p);
      if (iou > best.iou) best = { label: p, iou };
    }
    out.set(t, best);
  }
  return out;
}

const GAUSSIAN_FLOATS = 12;

/**
 * instancias.json (plan §3.3). Gaussian index is the invariant id; labels are
 * serialised separately as etiquetas.u32 (see labelsToBytes).
 */
export function buildInstancesJson({ escena, fecha, fuente, metodo, labels, gaussians = null, names = [], colors = [], views = [], embeddings = null }) {
  if (!(labels instanceof Uint32Array)) throw new Error("labels must be a Uint32Array");
  if (gaussians && gaussians.length !== labels.length * GAUSSIAN_FLOATS) {
    throw new Error("gaussians must have 12 floats per label");
  }
  const counts = new Map();
  const min = new Map();
  const max = new Map();
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i];
    if (!l) continue;
    counts.set(l, (counts.get(l) || 0) + 1);
    if (gaussians) {
      const o = i * GAUSSIAN_FLOATS;
      const mn = min.get(l) || [Infinity, Infinity, Infinity];
      const mx = max.get(l) || [-Infinity, -Infinity, -Infinity];
      for (let a = 0; a < 3; a++) {
        const v = gaussians[o + a];
        if (v < mn[a]) mn[a] = v;
        if (v > mx[a]) mx[a] = v;
      }
      min.set(l, mn);
      max.set(l, mx);
    }
  }
  const instancias = [...counts.keys()].sort((a, b) => a - b).map((l) => ({
    id_instancia: l,
    nombre: (names[l] && names[l].nombre) || (typeof names[l] === "string" ? names[l] : "") || `objeto ${l}`,
    nombre_es: (names[l] && names[l].nombre_es) || (typeof names[l] === "string" ? names[l] : "") || `objeto ${l}`,
    categoria: (names[l] && names[l].categoria) || "",
    confianza: names[l] && Number.isFinite(names[l].confianza) ? names[l].confianza : null,
    n_gaussianas: counts.get(l),
    bbox: gaussians ? { min: min.get(l), max: max.get(l) } : null,
    color: colors[l] || null,
    vistas: views.filter((v) => v.instancias && v.instancias.includes(l)).map((v) => v.indice),
    embedding_clip: embeddings && embeddings.vectors && embeddings.vectors[l]
      ? Array.from(embeddings.vectors[l], (v) => Math.round(v * 1e4) / 1e4)
      : null,
  }));
  return {
    version: 1,
    escena,
    fecha,
    fuente: { formato: fuente.formato || "", n_gaussianas: labels.length, sh_grado: fuente.sh_grado ?? 0, hash: fuente.hash || "" },
    metodo: {
      mascaras: metodo.mascaras || "",
      levantamiento: "flashsplat",
      sesgo_fondo: metodo.sesgo_fondo,
      umbral_iou: metodo.umbral_iou,
      difusion_iter: metodo.difusion_iter ?? 0,
      vistas: views.length,
      k_buffer: metodo.k_buffer ?? null,
    },
    n_instancias: instancias.length,
    embeddings: embeddings ? { modelo: embeddings.modelo || "", dimension: embeddings.dimension || 0 } : null,
    instancias,
  };
}

/** etiquetas.u32: little-endian u32 per gaussian. */
export function labelsToBytes(labels) {
  const out = new Uint8Array(labels.length * 4);
  const view = new DataView(out.buffer);
  for (let i = 0; i < labels.length; i++) view.setUint32(i * 4, labels[i], true);
  return out;
}

/** Inverse of labelsToBytes. */
export function labelsFromBytes(bytes) {
  if (bytes.byteLength % 4 !== 0) throw new Error("etiquetas.u32 length must be a multiple of 4");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Uint32Array(bytes.byteLength / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint32(i * 4, true);
  return out;
}
