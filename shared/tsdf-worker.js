/**
 * Web Worker for plan F6: fuse orbit depth maps into a TSDF and extract the
 * mesh off the main thread. Message: { id, views:[{depth, alpha, color?, width,
 * height, view, fx, fy, cx, cy}], center, radius, resolution, truncation,
 * carve, alphaMin, repair } → { id, ok, mesh, stats, ms }.
 */
import { TsdfVolume, largestComponent, meshStats } from "./tsdf.js";
import { repairMesh, validateMesh } from "./mesh-ops.js";

self.onmessage = (e) => {
  const d = e.data || {};
  const t0 = performance.now();
  try {
    const vol = new TsdfVolume({ center: d.center, radius: d.radius, resolution: d.resolution, truncation: d.truncation || null, margin: d.margin || 1.15 });
    let updated = 0;
    for (const v of d.views) updated += vol.integrate(v, { alphaMin: d.alphaMin ?? 0.05, carve: d.carve !== false });
    const tFuse = performance.now();
    const raw = d.printMode ? vol.extractMarchingTetrahedra() : vol.extract();
    const component = d.keepAll ? { ...raw, components: null, removedTriangles: 0 } : largestComponent(raw);
    const repaired = d.repair === false
      ? { mesh: component, before: validateMesh(component), after: validateMesh(component), repair: null }
      : repairMesh(component, { fillHoles: d.fillHoles !== false, maxHoleEdges: d.maxHoleEdges ?? 64 });
    const mesh = repaired.mesh;
    const stats = meshStats(mesh);
    const ms = performance.now() - t0;
    self.postMessage(
      {
        id: d.id,
        ok: true,
        mesh,
        stats: {
          ...stats,
          components: mesh.components,
          removedTriangles: mesh.removedTriangles,
          flipped: raw.flipped,
          updated,
          voxels: vol.n,
          voxelSize: vol.voxel,
          truncation: vol.truncation,
          validationBeforeRepair: repaired.before,
          validation: repaired.after,
          repair: repaired.repair,
        },
        ms,
        msFuse: tFuse - t0,
      },
      [mesh.positions.buffer, mesh.normals.buffer, mesh.colors.buffer, mesh.indices.buffer]
    );
  } catch (err) {
    self.postMessage({ id: d.id, ok: false, error: err && err.message ? err.message : String(err) });
  }
};
