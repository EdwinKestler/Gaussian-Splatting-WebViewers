/**
 * F4 naming helpers (plan §3.2.D): frame one instance for an isolated render,
 * apply the sidecar's names to the instance registry, and search instances by
 * text. Pure JS (no DOM, no WebGPU) so Node tests can cover it.
 */

const GAUSSIAN_FLOATS = 12;

/** Axis-aligned bounds of the gaussians carrying `label`; null when none. */
export function instanceBounds(labels, gaussians, label) {
  if (gaussians.length !== labels.length * GAUSSIAN_FLOATS) throw new Error("gaussians must have 12 floats per label");
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  let count = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== label) continue;
    const o = i * GAUSSIAN_FLOATS;
    for (let a = 0; a < 3; a++) {
      const v = gaussians[o + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
    count++;
  }
  if (!count) return null;
  const center = [0, 1, 2].map((a) => 0.5 * (min[a] + max[a]));
  const radius = Math.max(1e-3, 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]));
  return { min, max, center, radius, count };
}

/**
 * Orbit-camera target and distance that frame a bounding sphere in a view of
 * vertical fov `fov` (radians) with `margin` (1.0 = touching, 1.4 = comfortable).
 */
export function frameBounds(bounds, fov, margin = 1.4) {
  if (!(fov > 0 && fov < Math.PI)) throw new Error(`fov must be in (0, π), got ${fov}`);
  const distance = (bounds.radius * margin) / Math.sin(fov / 2);
  return { target: bounds.center.slice(), radius: Math.max(distance, bounds.radius * 1.05) };
}

/** Normalise text for matching: lowercase, no accents, collapsed spaces. */
export function normalizeText(text) {
  return String(text || "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Score an instance record against a free-text query (0 = no match).
 * Whole-field matches score higher than substrings; every query word must hit.
 * @param {{nombre?:string, nombre_es?:string, categoria?:string, descripcion_es?:string, name?:string}} record
 */
export function matchScore(record, query) {
  const q = normalizeText(query);
  if (!q) return 0;
  const fields = [
    [record.nombre_es, 3],
    [record.nombre, 3],
    [record.name, 3],
    [record.categoria, 2],
    [record.descripcion_es, 1],
  ].map(([v, w]) => [normalizeText(v), w]).filter(([v]) => v);
  const words = q.split(" ");
  let score = 0;
  for (const word of words) {
    let best = 0;
    for (const [value, weight] of fields) {
      if (value === word) best = Math.max(best, weight * 2);
      else if (value.split(" ").includes(word)) best = Math.max(best, weight * 1.5);
      else if (value.includes(word)) best = Math.max(best, weight);
    }
    if (!best) return 0;
    score += best;
  }
  return score;
}

/**
 * Instances matching `query`, best first.
 * @param {Array<{label:number}>} records any objects with a `label` and name fields
 * @returns {Array<{label:number, score:number}>}
 */
export function searchInstances(records, query) {
  return records
    .map((r) => ({ label: r.label, score: matchScore(r, query) }))
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score || a.label - b.label);
}

/**
 * Merge sidecar name results into a registry Map(label → entry). Entries gain
 * nombre, nombre_es, categoria, confianza, descripcion_es and `name` (display
 * name = nombre_es). Failed results (ok:false) keep the previous name.
 * @returns {{applied:number, failed:number}}
 */
export function applyNames(entries, results) {
  let applied = 0;
  let failed = 0;
  for (const r of results || []) {
    const label = Number(r.id_instancia);
    const entry = entries.get(label);
    if (!entry) continue;
    if (r.ok === false) {
      failed++;
      entry.error = r.error || "sin nombre";
      continue;
    }
    entry.nombre = r.nombre || entry.nombre || entry.name;
    entry.nombre_es = r.nombre_es || entry.nombre;
    entry.categoria = r.categoria || "otro";
    entry.confianza = Number.isFinite(r.confianza) ? r.confianza : null;
    entry.descripcion_es = r.descripcion_es || "";
    entry.name = entry.nombre_es;
    delete entry.error;
    applied++;
  }
  return { applied, failed };
}
