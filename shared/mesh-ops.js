/**
 * Mesh checks and conservative repairs used by the 3D-print export path.
 * A mesh is { positions, indices, normals?, colors? } with triangle indices.
 */

const finite3 = (p, i) => Number.isFinite(p[i]) && Number.isFinite(p[i + 1]) && Number.isFinite(p[i + 2]);
const edgeKey = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);

function boundsOf(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  let finiteVertices = 0;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    if (!finite3(positions, i)) continue;
    finiteVertices++;
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], positions[i + a]);
      max[a] = Math.max(max[a], positions[i + a]);
    }
  }
  if (!finiteVertices) return { min: [0, 0, 0], max: [0, 0, 0], size: [0, 0, 0], diagonal: 0 };
  const size = max.map((v, a) => v - min[a]);
  return { min, max, size, diagonal: Math.hypot(...size) };
}

function triangleGeometry(positions, a, b, c) {
  const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
  const bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
  const cx = positions[c * 3], cy = positions[c * 3 + 1], cz = positions[c * 3 + 2];
  const ux = bx - ax, uy = by - ay, uz = bz - az;
  const vx = cx - ax, vy = cy - ay, vz = cz - az;
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  return {
    area2: Math.hypot(nx, ny, nz),
    signedVolume6: ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx),
  };
}

function topology(positions, indices, areaEpsilon) {
  const nv = Math.floor(positions.length / 3);
  const finite = new Uint8Array(nv);
  let nonFiniteVertices = 0;
  for (let v = 0; v < nv; v++) {
    finite[v] = finite3(positions, v * 3) ? 1 : 0;
    if (!finite[v]) nonFiniteVertices++;
  }
  const edges = new Map();
  const seen = new Set();
  let invalidIndexTriangles = 0, degenerateTriangles = 0, duplicateTriangles = 0;
  let surfaceArea = 0, signedVolume = 0, usableTriangles = 0;
  const triCount = Math.floor(indices.length / 3);
  for (let t = 0; t < triCount; t++) {
    const a = Number(indices[t * 3]), b = Number(indices[t * 3 + 1]), c = Number(indices[t * 3 + 2]);
    if (![a, b, c].every((v) => Number.isInteger(v) && v >= 0 && v < nv && finite[v])) {
      invalidIndexTriangles++;
      continue;
    }
    if (a === b || b === c || c === a) { degenerateTriangles++; continue; }
    const g = triangleGeometry(positions, a, b, c);
    if (!(g.area2 > areaEpsilon)) { degenerateTriangles++; continue; }
    const faceKey = [a, b, c].sort((x, y) => x - y).join(":");
    if (seen.has(faceKey)) { duplicateTriangles++; continue; }
    seen.add(faceKey);
    usableTriangles++;
    surfaceArea += g.area2 / 2;
    signedVolume += g.signedVolume6 / 6;
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const key = edgeKey(u, v);
      const list = edges.get(key) || [];
      list.push({ triangle: t, direction: u < v ? 1 : -1, from: u, to: v });
      edges.set(key, list);
    }
  }
  let boundaryEdges = 0, nonManifoldEdges = 0, inconsistentWindingEdges = 0;
  for (const uses of edges.values()) {
    if (uses.length === 1) boundaryEdges++;
    else if (uses.length > 2) nonManifoldEdges++;
    else if (uses[0].direction === uses[1].direction) inconsistentWindingEdges++;
  }
  return {
    edges,
    summary: {
      vertices: nv,
      triangles: triCount,
      usableTriangles,
      nonFiniteVertices,
      invalidIndexTriangles,
      degenerateTriangles,
      duplicateTriangles,
      boundaryEdges,
      nonManifoldEdges,
      inconsistentWindingEdges,
      surfaceArea,
      signedVolume,
    },
  };
}

/** Return geometry and edge-manifold diagnostics without mutating the mesh. */
export function validateMesh(mesh, { areaEpsilon = null } = {}) {
  if (!mesh || !mesh.positions || !mesh.indices) throw new Error("mesh requires positions and indices");
  if (mesh.positions.length % 3) throw new Error("positions length must be divisible by 3");
  const malformedIndexTail = mesh.indices.length % 3;
  const bbox = boundsOf(mesh.positions);
  const eps = areaEpsilon ?? Math.max(1e-20, bbox.diagonal * bbox.diagonal * 1e-12);
  const { summary } = topology(mesh.positions, mesh.indices, eps);
  const geometryValid = !malformedIndexTail && !summary.nonFiniteVertices && !summary.invalidIndexTriangles && !summary.degenerateTriangles && !summary.duplicateTriangles && summary.usableTriangles > 0;
  const manifold = !summary.boundaryEdges && !summary.nonManifoldEdges;
  const oriented = !summary.inconsistentWindingEdges;
  return {
    ...summary,
    malformedIndexTail,
    bbox,
    areaEpsilon: eps,
    geometryValid,
    manifold,
    oriented,
    watertight: geometryValid && manifold && oriented,
    printable: geometryValid && manifold && oriented && Math.abs(summary.signedVolume) > eps,
  };
}

function orientTriangles(triangles) {
  const edgeUses = new Map();
  for (let t = 0; t < triangles.length; t++) {
    const tri = triangles[t];
    for (let s = 0; s < 3; s++) {
      const a = tri[s], b = tri[(s + 1) % 3];
      const key = edgeKey(a, b);
      const list = edgeUses.get(key) || [];
      list.push({ triangle: t, direction: a < b ? 1 : -1 });
      edgeUses.set(key, list);
    }
  }
  const neighbors = Array.from({ length: triangles.length }, () => []);
  for (const uses of edgeUses.values()) {
    if (uses.length !== 2) continue;
    const [a, b] = uses;
    const oppositeFlip = a.direction === b.direction ? 1 : 0;
    neighbors[a.triangle].push([b.triangle, oppositeFlip]);
    neighbors[b.triangle].push([a.triangle, oppositeFlip]);
  }
  const flip = new Int8Array(triangles.length).fill(-1);
  const components = [];
  let conflicts = 0;
  for (let seed = 0; seed < triangles.length; seed++) {
    if (flip[seed] >= 0) continue;
    flip[seed] = 0;
    const component = [];
    const queue = [seed];
    for (let qi = 0; qi < queue.length; qi++) {
      const t = queue[qi];
      component.push(t);
      for (const [n, relation] of neighbors[t]) {
        const wanted = flip[t] ^ relation;
        if (flip[n] < 0) { flip[n] = wanted; queue.push(n); }
        else if (flip[n] !== wanted) conflicts++;
      }
    }
    components.push(component);
  }
  let flipped = 0;
  for (let t = 0; t < triangles.length; t++) {
    if (!flip[t]) continue;
    [triangles[t][1], triangles[t][2]] = [triangles[t][2], triangles[t][1]];
    flipped++;
  }
  return { components, flipped, conflicts };
}

function boundaryLoops(triangles) {
  const edges = new Map();
  for (let t = 0; t < triangles.length; t++) {
    const tri = triangles[t];
    for (let s = 0; s < 3; s++) {
      const from = tri[s], to = tri[(s + 1) % 3];
      const key = edgeKey(from, to);
      const uses = edges.get(key) || [];
      uses.push({ from, to });
      edges.set(key, uses);
    }
  }
  const boundary = [...edges.values()].filter((uses) => uses.length === 1).map((uses) => uses[0]);
  const outgoing = new Map();
  const incoming = new Map();
  for (const e of boundary) {
    const out = outgoing.get(e.from) || []; out.push(e); outgoing.set(e.from, out);
    const inc = incoming.get(e.to) || []; inc.push(e); incoming.set(e.to, inc);
  }
  const used = new Set();
  const loops = [];
  const rejected = [];
  for (const first of boundary) {
    const firstKey = `${first.from}>${first.to}`;
    if (used.has(firstKey)) continue;
    const vertices = [first.from];
    let edge = first;
    let closed = false;
    while (edge && vertices.length <= boundary.length + 1) {
      const key = `${edge.from}>${edge.to}`;
      if (used.has(key)) break;
      used.add(key);
      vertices.push(edge.to);
      if (edge.to === vertices[0]) { closed = true; break; }
      const next = outgoing.get(edge.to) || [];
      edge = next.length === 1 && (incoming.get(edge.to) || []).length === 1 ? next[0] : null;
    }
    if (closed && vertices.length >= 4) loops.push(vertices.slice(0, -1));
    else rejected.push(vertices);
  }
  return { loops, rejected, boundaryEdges: boundary.length };
}

function compactMesh(positions, colors, triangles, useFloat64 = false) {
  const used = new Uint8Array(positions.length / 3);
  for (const tri of triangles) for (const v of tri) used[v] = 1;
  const remap = new Int32Array(used.length).fill(-1);
  const outP = [], outC = [];
  for (let v = 0; v < used.length; v++) {
    if (!used[v]) continue;
    remap[v] = outP.length / 3;
    outP.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
    outC.push(colors[v * 3] ?? 0.7, colors[v * 3 + 1] ?? 0.7, colors[v * 3 + 2] ?? 0.7);
  }
  const indices = [];
  for (const tri of triangles) indices.push(remap[tri[0]], remap[tri[1]], remap[tri[2]]);
  const p = useFloat64 ? Float64Array.from(outP) : Float32Array.from(outP);
  const normals = new Float32Array(p.length);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
    const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const v of [a, b, c]) { normals[v * 3] += nx; normals[v * 3 + 1] += ny; normals[v * 3 + 2] += nz; }
  }
  for (let v = 0; v < normals.length / 3; v++) {
    const n = Math.hypot(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]) || 1;
    normals[v * 3] /= n; normals[v * 3 + 1] /= n; normals[v * 3 + 2] /= n;
  }
  return {
    positions: p,
    normals,
    colors: Float32Array.from(outC),
    indices: Uint32Array.from(indices),
    vertexCount: p.length / 3,
    triangleCount: indices.length / 3,
  };
}

/**
 * Repair only operations with an unambiguous local interpretation. Large or
 * branching boundary holes and non-manifold edges remain visible to validation.
 */
export function repairMesh(mesh, { weldTolerance = null, fillHoles = true, maxHoleEdges = 64, areaEpsilon = null, _pass = 0 } = {}) {
  const before = validateMesh(mesh, { areaEpsilon });
  const tolerance = weldTolerance ?? Math.max(1e-9, before.bbox.diagonal * 1e-6);
  if (!(tolerance > 0)) throw new Error("weldTolerance must be > 0");
  if (!Number.isInteger(maxHoleEdges) || maxHoleEdges < 3) throw new Error("maxHoleEdges must be an integer >= 3");
  const positions = [];
  const colorSums = [];
  const colorCounts = [];
  const remap = new Int32Array(before.vertices).fill(-1);
  const buckets = new Map();
  for (let v = 0; v < before.vertices; v++) {
    const i = v * 3;
    if (!finite3(mesh.positions, i)) continue;
    const q = [Math.floor(mesh.positions[i] / tolerance), Math.floor(mesh.positions[i + 1] / tolerance), Math.floor(mesh.positions[i + 2] / tolerance)];
    let out = null;
    for (let dz = -1; dz <= 1 && out == null; dz++) {
      for (let dy = -1; dy <= 1 && out == null; dy++) {
        for (let dx = -1; dx <= 1 && out == null; dx++) {
          const candidates = buckets.get(`${q[0] + dx}:${q[1] + dy}:${q[2] + dz}`) || [];
          for (const candidate of candidates) {
            const distance = Math.hypot(
              positions[candidate * 3] - mesh.positions[i],
              positions[candidate * 3 + 1] - mesh.positions[i + 1],
              positions[candidate * 3 + 2] - mesh.positions[i + 2]
            );
            if (distance <= tolerance) { out = candidate; break; }
          }
        }
      }
    }
    if (out == null) {
      out = positions.length / 3;
      const key = `${q[0]}:${q[1]}:${q[2]}`;
      const bucket = buckets.get(key) || [];
      bucket.push(out); buckets.set(key, bucket);
      positions.push(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]);
      colorSums.push(0, 0, 0); colorCounts.push(0);
    }
    remap[v] = out;
    if (mesh.colors && mesh.colors.length >= i + 3 && finite3(mesh.colors, i)) {
      colorSums[out * 3] += mesh.colors[i]; colorSums[out * 3 + 1] += mesh.colors[i + 1]; colorSums[out * 3 + 2] += mesh.colors[i + 2]; colorCounts[out]++;
    }
  }
  const colors = [];
  for (let v = 0; v < positions.length / 3; v++) {
    const n = colorCounts[v] || 1;
    colors.push(colorCounts[v] ? colorSums[v * 3] / n : 0.7, colorCounts[v] ? colorSums[v * 3 + 1] / n : 0.7, colorCounts[v] ? colorSums[v * 3 + 2] / n : 0.7);
  }
  const triangles = [];
  const seen = new Set();
  let removedInvalid = mesh.indices.length % 3 ? 1 : 0, removedDegenerate = 0, removedDuplicates = 0;
  // Use a margin above the reporting threshold so the float32 compacted mesh
  // cannot move a borderline face back below that threshold through rounding.
  const eps = areaEpsilon ?? before.areaEpsilon * 4;
  for (let t = 0; t + 2 < mesh.indices.length; t += 3) {
    const original = [mesh.indices[t], mesh.indices[t + 1], mesh.indices[t + 2]];
    if (!original.every((v) => Number.isInteger(v) && v >= 0 && v < remap.length && remap[v] >= 0)) { removedInvalid++; continue; }
    const tri = original.map((v) => remap[v]);
    if (new Set(tri).size < 3 || !(triangleGeometry(positions, ...tri).area2 > eps)) { removedDegenerate++; continue; }
    const key = [...tri].sort((a, b) => a - b).join(":");
    if (seen.has(key)) { removedDuplicates++; continue; }
    seen.add(key); triangles.push(tri);
  }
  const oriented = orientTriangles(triangles);
  let holesFound = 0, holesFilled = 0, holeEdgesFilled = 0, rejectedBoundaryChains = 0;
  if (fillHoles) {
    const boundary = boundaryLoops(triangles);
    holesFound = boundary.loops.length;
    rejectedBoundaryChains = boundary.rejected.length;
    for (const loop of boundary.loops) {
      if (loop.length > maxHoleEdges) continue;
      const center = [0, 0, 0], color = [0, 0, 0];
      for (const v of loop) {
        for (let a = 0; a < 3; a++) { center[a] += positions[v * 3 + a]; color[a] += colors[v * 3 + a]; }
      }
      const centerIndex = positions.length / 3;
      for (let a = 0; a < 3; a++) { positions.push(center[a] / loop.length); colors.push(color[a] / loop.length); }
      for (let i = 0; i < loop.length; i++) triangles.push([loop[(i + 1) % loop.length], loop[i], centerIndex]);
      holesFilled++; holeEdgesFilled += loop.length;
    }
  }
  const finalOrientation = orientTriangles(triangles);
  let outwardFlippedComponents = 0;
  for (const component of finalOrientation.components) {
    let volume = 0;
    for (const t of component) volume += triangleGeometry(positions, ...triangles[t]).signedVolume6 / 6;
    if (volume >= 0) continue;
    for (const t of component) [triangles[t][1], triangles[t][2]] = [triangles[t][2], triangles[t][1]];
    outwardFlippedComponents++;
  }
  const repaired = compactMesh(positions, colors, triangles, mesh.positions instanceof Float64Array);
  repaired.components = mesh.components ?? finalOrientation.components.length;
  repaired.removedTriangles = mesh.removedTriangles ?? 0;
  const after = validateMesh(repaired, { areaEpsilon });
  const repair = {
    weldTolerance: tolerance,
    cleanupAreaEpsilon: eps,
    weldedVertices: before.vertices - positions.length / 3 + holesFilled,
    removedInvalidTriangles: removedInvalid,
    removedDegenerateTriangles: removedDegenerate,
    removedDuplicateTriangles: removedDuplicates,
    windingFlips: oriented.flipped + finalOrientation.flipped,
    windingConflicts: oriented.conflicts + finalOrientation.conflicts,
    holesFound,
    holesFilled,
    holeEdgesFilled,
    rejectedBoundaryChains,
    outwardFlippedComponents,
  };
  if (!after.geometryValid && _pass < 2) {
    const refined = repairMesh(repaired, {
      weldTolerance: tolerance,
      fillHoles,
      maxHoleEdges,
      areaEpsilon: after.areaEpsilon * 4,
      _pass: _pass + 1,
    });
    return {
      mesh: refined.mesh,
      before,
      after: validateMesh(refined.mesh),
      repair: {
        ...repair,
        refinementPasses: 1 + (refined.repair.refinementPasses || 0),
        refinement: refined.repair,
      },
    };
  }
  return {
    mesh: repaired,
    before,
    after,
    repair,
  };
}

/** Convert arbitrary scene units to millimetres and place the mesh on z=0. */
export function scaleMeshToMillimeters(mesh, { maxDimensionMm = 100 } = {}) {
  if (!(maxDimensionMm > 0) || !Number.isFinite(maxDimensionMm)) throw new Error("maxDimensionMm must be finite and > 0");
  const source = validateMesh(mesh);
  const longest = Math.max(...source.bbox.size);
  if (!(longest > 0)) throw new Error("mesh has no measurable extent");
  const scale = maxDimensionMm / longest;
  // Keep the scale transform in float64. A dense TSDF mesh can contain very
  // short valid edges; rounding the transformed coordinates back to float32
  // can collapse those edges and create zero-area triangles before XML output.
  const positions = new Float64Array(mesh.positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] = (mesh.positions[i] - source.bbox.min[0]) * scale;
    positions[i + 1] = (mesh.positions[i + 1] - source.bbox.min[1]) * scale;
    positions[i + 2] = (mesh.positions[i + 2] - source.bbox.min[2]) * scale;
  }
  const out = {
    ...mesh,
    positions,
    normals: mesh.normals ? new Float32Array(mesh.normals) : undefined,
    colors: mesh.colors ? new Float32Array(mesh.colors) : undefined,
    indices: new Uint32Array(mesh.indices),
    vertexCount: positions.length / 3,
    triangleCount: mesh.indices.length / 3,
  };
  const result = validateMesh(out);
  return { mesh: out, scale, unit: "millimeter", maxDimensionMm, sourceBbox: source.bbox, bboxMm: result.bbox, validation: result };
}
