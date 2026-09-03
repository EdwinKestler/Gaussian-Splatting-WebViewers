/**
 * Module worker for shared/graph.js (plan F2): builds the superpoint graph off
 * the main thread. Messages:
 *   { id, type: "build", gaussians: Float32Array(N*12), colors: Float32Array(N*3)|null, options }
 *     → { id, ok: true, type: "build", result } with the typed arrays transferred
 *   anything else → { id, ok: false, error }
 * The caller keeps its own copy of the cloud (post a copy, not the renderer's arrays).
 */
import { buildSuperpointGraph } from "./graph.js";

function errorText(err) {
  return err && err.message ? err.message : String(err);
}

self.onmessage = (event) => {
  const data = event.data || {};
  const id = data.id;
  try {
    if (data.type !== "build") throw new Error(`tipo de mensaje desconocido: ${data.type}`);
    const t0 = performance.now();
    const result = buildSuperpointGraph(data.gaussians, data.colors || null, data.options || {});
    result.stats.msWorker = performance.now() - t0;
    const transfer = [
      result.csr.offsets.buffer,
      result.csr.neighbors.buffer,
      result.csr.dist2.buffer,
      result.csr.weights.buffer,
      result.superpoint.buffer,
      result.sizes.buffer,
      result.centroids.buffer,
    ];
    self.postMessage({ id, ok: true, type: "build", result }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, type: data.type || "", error: errorText(err) });
  }
};
