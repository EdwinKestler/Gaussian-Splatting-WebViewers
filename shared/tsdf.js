/**
 * Plan F6 — mesh per instance from orbit depth maps, entirely in JS:
 *   TsdfVolume.integrate(view)  fuses one depth (+ colour) map into a truncated
 *                               signed distance volume (Curless–Levoy / KinectFusion)
 *   TsdfVolume.extract()        naive surface nets (one vertex per sign-changing
 *                               voxel, quads across sign-changing edges) → triangles
 *   largestComponent(mesh)      drops floating fragments
 *   orbitCameras(...)           Fibonacci-sphere cameras around a bounding sphere
 *
 * Camera convention (same as gpu-renderer.js renderOffscreen): `view` is a
 * column-major world→camera matrix, the camera looks down -z, depth is the
 * positive view-space z, pixel (u, v) = (cx + fx·x/z, cy − fy·y/z) with the
 * origin at the top-left. Depth maps are Float32Array(width·height) with an
 * `alpha` coverage map (0 = empty ray).
 */

export const DEFAULT_RESOLUTION = 96;

// ---------------------------------------------------------------- cameras

/** Column-major lookAt (world → camera), camera looking down -z. */
export function lookAtMatrix(eye, target, up = [0, 1, 0]) {
  let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let n = Math.hypot(zx, zy, zz) || 1;
  zx /= n; zy /= n; zz /= n;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  n = Math.hypot(xx, xy, xz) || 1;
  xx /= n; xy /= n; xz /= n;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

/** Directions on the unit sphere (Fibonacci lattice), evenly spread. */
export function fibonacciDirections(count) {
  const out = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < count; i++) {
    const y = 1 - (2 * (i + 0.5)) / count;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const t = golden * i;
    out.push([Math.cos(t) * r, y, Math.sin(t) * r]);
  }
  return out;
}

/**
 * Pinhole cameras on a sphere of `distance` around `center`, all looking at it.
 * `fov` is the vertical field of view (radians); fx follows `aspect` (W/H of
 * the field of view, not necessarily width/height of the image).
 */
export function orbitCameras({ center, distance, count = 24, fov = (50 * Math.PI) / 180, width, height, aspect = width / height, maxPitch = 1.3 }) {
  const fy = height / (2 * Math.tan(fov / 2));
  const fx = (width / (2 * Math.tan(fov / 2))) / aspect;
  return fibonacciDirections(count).map((d) => {
    // Keep the camera off the poles so the up vector stays valid.
    const pitch = Math.max(-maxPitch, Math.min(maxPitch, Math.asin(d[1])));
    const yaw = Math.atan2(d[0], d[2]);
    const dir = [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
    const eye = [center[0] + dir[0] * distance, center[1] + dir[1] * distance, center[2] + dir[2] * distance];
    return { eye, yaw, pitch, view: lookAtMatrix(eye, center), fx, fy, cx: width / 2, cy: height / 2, width, height };
  });
}

// ---------------------------------------------------------------- volume

export class TsdfVolume {
  /**
   * @param {{center:number[], radius:number, resolution?:number, margin?:number, truncation?:number}} o
   *   A cube of side 2·radius·margin centred on `center`, `resolution` voxels per side;
   *   `truncation` in world units (default 3 voxels).
   */
  constructor({ center, radius, resolution = DEFAULT_RESOLUTION, margin = 1.15, truncation = null }) {
    if (!(radius > 0)) throw new Error("radius must be > 0");
    this.n = Math.max(8, resolution | 0);
    this.size = 2 * radius * margin;
    this.voxel = this.size / (this.n - 1);
    this.origin = [center[0] - this.size / 2, center[1] - this.size / 2, center[2] - this.size / 2];
    this.truncation = truncation || this.voxel * 3;
    const total = this.n * this.n * this.n;
    this.tsdf = new Float32Array(total).fill(1);
    this.weight = new Float32Array(total);
    this.color = new Float32Array(total * 3);
    this.colorWeight = new Float32Array(total);
    this.views = 0;
  }

  index(i, j, k) {
    return (k * this.n + j) * this.n + i;
  }

  /** World position of a sample. */
  position(i, j, k) {
    return [this.origin[0] + i * this.voxel, this.origin[1] + j * this.voxel, this.origin[2] + k * this.voxel];
  }

  /**
   * Fuse one view. `depth`/`alpha` are per pixel; `color` (optional) is RGBA8.
   * Rays with alpha < alphaMin see nothing: with `carve` they push the volume
   * towards "empty" with a small weight, which removes floating blobs.
   */
  integrate({ depth, alpha = null, color = null, width, height, view, fx, fy, cx, cy }, { alphaMin = 0.05, carve = true, carveWeight = 0.3 } = {}) {
    if (!depth || depth.length !== width * height) throw new Error("depth must have width*height values");
    const n = this.n;
    const tr = this.truncation;
    const m = view;
    const { tsdf, weight, color: col, colorWeight } = this;
    let updated = 0;
    for (let k = 0; k < n; k++) {
      const wz = this.origin[2] + k * this.voxel;
      for (let j = 0; j < n; j++) {
        const wy = this.origin[1] + j * this.voxel;
        for (let i = 0; i < n; i++) {
          const wx = this.origin[0] + i * this.voxel;
          const cz = m[2] * wx + m[6] * wy + m[10] * wz + m[14];
          const z = -cz;
          if (z <= 1e-6) continue;
          const cxv = m[0] * wx + m[4] * wy + m[8] * wz + m[12];
          const cyv = m[1] * wx + m[5] * wy + m[9] * wz + m[13];
          const u = Math.round(cx + (fx * cxv) / z);
          const v = Math.round(cy - (fy * cyv) / z);
          if (u < 0 || v < 0 || u >= width || v >= height) continue;
          const p = v * width + u;
          const a = alpha ? alpha[p] : 1;
          const idx = (k * n + j) * n + i;
          if (a < alphaMin) {
            if (!carve) continue;
            const w0 = weight[idx];
            tsdf[idx] = (tsdf[idx] * w0 + 1 * carveWeight) / (w0 + carveWeight);
            weight[idx] = w0 + carveWeight;
            continue;
          }
          const sdf = depth[p] - z;
          if (sdf < -tr) continue; // behind the surface: unobserved
          const d = Math.min(1, sdf / tr);
          const w = a;
          const w0 = weight[idx];
          tsdf[idx] = (tsdf[idx] * w0 + d * w) / (w0 + w);
          weight[idx] = w0 + w;
          updated++;
          if (color && Math.abs(sdf) < tr * 0.6) {
            const cw = colorWeight[idx];
            const c = idx * 3;
            col[c] = (col[c] * cw + color[p * 4] * w) / (cw + w);
            col[c + 1] = (col[c + 1] * cw + color[p * 4 + 1] * w) / (cw + w);
            col[c + 2] = (col[c + 2] * cw + color[p * 4 + 2] * w) / (cw + w);
            colorWeight[idx] = cw + w;
          }
        }
      }
    }
    this.views++;
    return updated;
  }

  /**
   * Naive surface nets over the fused volume.
   * @returns {{positions:Float32Array, normals:Float32Array, colors:Float32Array, indices:Uint32Array, vertexCount:number, triangleCount:number}}
   */
  extract({ minWeight = 0.5 } = {}) {
    const n = this.n;
    const { tsdf, weight, color, colorWeight } = this;
    const cubeIndex = new Int32Array((n - 1) * (n - 1) * (n - 1)).fill(-1);
    const cubeId = (i, j, k) => (k * (n - 1) + j) * (n - 1) + i;
    const positions = [];
    const normals = [];
    const colors = [];
    const corner = new Float32Array(8);
    const cornerIdx = new Int32Array(8);
    // corner c has offset bit0 → +i, bit1 → +j, bit2 → +k
    for (let k = 0; k < n - 1; k++) {
      for (let j = 0; j < n - 1; j++) {
        for (let i = 0; i < n - 1; i++) {
          let mask = 0;
          let observed = true;
          for (let c = 0; c < 8; c++) {
            const idx = this.index(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
            cornerIdx[c] = idx;
            if (weight[idx] < minWeight) { observed = false; break; }
            corner[c] = tsdf[idx];
            if (corner[c] < 0) mask |= 1 << c;
          }
          if (!observed || mask === 0 || mask === 0xff) continue;
          // Vertex: mean of the edge crossings.
          let sx = 0, sy = 0, sz = 0, cnt = 0;
          for (let e = 0; e < 12; e++) {
            const [a, b] = CUBE_EDGES[e];
            if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
            const va = corner[a], vb = corner[b];
            const t = va / (va - vb);
            const ax = a & 1, ay = (a >> 1) & 1, az = (a >> 2) & 1;
            const bx = b & 1, by = (b >> 1) & 1, bz = (b >> 2) & 1;
            sx += ax + (bx - ax) * t;
            sy += ay + (by - ay) * t;
            sz += az + (bz - az) * t;
            cnt++;
          }
          const vx = i + sx / cnt, vy = j + sy / cnt, vz = k + sz / cnt;
          const vid = positions.length / 3;
          positions.push(this.origin[0] + vx * this.voxel, this.origin[1] + vy * this.voxel, this.origin[2] + vz * this.voxel);
          // Normal: gradient of the trilinear field (outward: tsdf grows outside).
          let gx = 0, gy = 0, gz = 0;
          for (let c = 0; c < 8; c++) {
            gx += (c & 1 ? 1 : -1) * corner[c];
            gy += ((c >> 1) & 1 ? 1 : -1) * corner[c];
            gz += ((c >> 2) & 1 ? 1 : -1) * corner[c];
          }
          const gl = Math.hypot(gx, gy, gz) || 1;
          normals.push(gx / gl, gy / gl, gz / gl);
          // Colour: weighted mean of the observed corner colours.
          let cr = 0, cg = 0, cb = 0, cw = 0;
          for (let c = 0; c < 8; c++) {
            const w = colorWeight[cornerIdx[c]];
            if (w <= 0) continue;
            cr += color[cornerIdx[c] * 3] * w;
            cg += color[cornerIdx[c] * 3 + 1] * w;
            cb += color[cornerIdx[c] * 3 + 2] * w;
            cw += w;
          }
          colors.push(cw ? cr / cw / 255 : 0.7, cw ? cg / cw / 255 : 0.7, cw ? cb / cw / 255 : 0.7);
          cubeIndex[cubeId(i, j, k)] = vid;
        }
      }
    }
    // Faces: for every axis edge leaving corner 0 of a cube with a sign change,
    // the four cubes around that edge form a quad.
    const indices = [];
    const sampleInside = (i, j, k) => tsdf[this.index(i, j, k)] < 0;
    for (let k = 1; k < n - 1; k++) {
      for (let j = 1; j < n - 1; j++) {
        for (let i = 1; i < n - 1; i++) {
          const v0 = cubeIndex[cubeId(i, j, k)];
          if (v0 < 0) continue;
          const inside0 = sampleInside(i, j, k);
          for (let axis = 0; axis < 3; axis++) {
            const di = axis === 0 ? 1 : 0, dj = axis === 1 ? 1 : 0, dk = axis === 2 ? 1 : 0;
            const inside1 = sampleInside(i + di, j + dj, k + dk);
            if (inside0 === inside1) continue;
            // the other two axes
            const ui = axis === 0 ? 0 : 1, uj = axis === 0 ? 1 : 0, uk = 0; // u: first other axis (i or j)
            const wi = 0, wj = axis === 2 ? 1 : 0, wk = axis === 2 ? 0 : 1; // w: second other axis (j or k)
            const v1 = cubeIndex[cubeId(i - ui, j - uj, k - uk)];
            const v2 = cubeIndex[cubeId(i - ui - wi, j - uj - wj, k - uk - wk)];
            const v3 = cubeIndex[cubeId(i - wi, j - wj, k - wk)];
            if (v1 < 0 || v2 < 0 || v3 < 0) continue;
            // Wind so the face normal points from inside (tsdf < 0) to outside;
            // (u, w) is cyclic for axes 0 and 2 but reversed for axis 1.
            if (inside0 !== (axis === 1)) indices.push(v0, v1, v2, v0, v2, v3);
            else indices.push(v0, v3, v2, v0, v2, v1);
          }
        }
      }
    }
    const mesh = {
      positions: Float32Array.from(positions),
      normals: Float32Array.from(normals),
      colors: Float32Array.from(colors),
      indices: Uint32Array.from(indices),
    };
    mesh.vertexCount = mesh.positions.length / 3;
    mesh.triangleCount = mesh.indices.length / 3;
    return fixWinding(mesh);
  }

  /**
   * Marching tetrahedra over the same TSDF. The fixed six-tetrahedra cube
   * decomposition shares the same face diagonals between neighbouring cells,
   * avoiding ambiguous surface-net junctions in meshes intended for printing.
   */
  extractMarchingTetrahedra({ minWeight = 0.5 } = {}) {
    const n = this.n;
    const { tsdf, weight, color, colorWeight } = this;
    const positions = [], normals = [], colors = [], indices = [];
    const edgeVertices = new Map();
    const tetrahedra = [
      [0, 5, 1, 7], [0, 1, 3, 7], [0, 3, 2, 7],
      [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7],
    ];
    const gradient = (i, j, k) => {
      const im = Math.max(0, i - 1), ip = Math.min(n - 1, i + 1);
      const jm = Math.max(0, j - 1), jp = Math.min(n - 1, j + 1);
      const km = Math.max(0, k - 1), kp = Math.min(n - 1, k + 1);
      return [
        tsdf[this.index(ip, j, k)] - tsdf[this.index(im, j, k)],
        tsdf[this.index(i, jp, k)] - tsdf[this.index(i, jm, k)],
        tsdf[this.index(i, j, kp)] - tsdf[this.index(i, j, km)],
      ];
    };
    for (let k = 0; k < n - 1; k++) {
      for (let j = 0; j < n - 1; j++) {
        for (let i = 0; i < n - 1; i++) {
          const corners = Array.from({ length: 8 }, (_, c) => {
            const x = i + (c & 1), y = j + ((c >> 1) & 1), z = k + ((c >> 2) & 1);
            const id = this.index(x, y, z);
            return { x, y, z, id, value: tsdf[id] };
          });
          if (corners.some((c) => weight[c.id] < minWeight)) continue;
          const crossing = (ca, cb) => {
            const a = corners[ca], b = corners[cb];
            const lo = Math.min(a.id, b.id), hi = Math.max(a.id, b.id);
            const key = `${lo}:${hi}`;
            if (edgeVertices.has(key)) return edgeVertices.get(key);
            const t = a.value / (a.value - b.value);
            const gx = a.x + (b.x - a.x) * t, gy = a.y + (b.y - a.y) * t, gz = a.z + (b.z - a.z) * t;
            const vertex = positions.length / 3;
            positions.push(this.origin[0] + gx * this.voxel, this.origin[1] + gy * this.voxel, this.origin[2] + gz * this.voxel);
            const ga = gradient(a.x, a.y, a.z), gb = gradient(b.x, b.y, b.z);
            const nx = ga[0] + (gb[0] - ga[0]) * t, ny = ga[1] + (gb[1] - ga[1]) * t, nz = ga[2] + (gb[2] - ga[2]) * t;
            const nl = Math.hypot(nx, ny, nz) || 1;
            normals.push(nx / nl, ny / nl, nz / nl);
            const aw = colorWeight[a.id], bw = colorWeight[b.id];
            const cw = aw * (1 - t) + bw * t;
            for (let axis = 0; axis < 3; axis++) {
              const value = cw > 0 ? (color[a.id * 3 + axis] * aw * (1 - t) + color[b.id * 3 + axis] * bw * t) / cw / 255 : 0.7;
              colors.push(value);
            }
            edgeVertices.set(key, vertex);
            return vertex;
          };
          for (const tet of tetrahedra) {
            const inside = tet.filter((c) => corners[c].value < 0);
            if (!inside.length || inside.length === 4) continue;
            const outside = tet.filter((c) => corners[c].value >= 0);
            if (inside.length === 1) {
              indices.push(crossing(inside[0], outside[0]), crossing(inside[0], outside[1]), crossing(inside[0], outside[2]));
            } else if (inside.length === 3) {
              indices.push(crossing(outside[0], inside[0]), crossing(outside[0], inside[1]), crossing(outside[0], inside[2]));
            } else {
              const a = crossing(inside[0], outside[0]), b = crossing(inside[0], outside[1]);
              const c = crossing(inside[1], outside[0]), d = crossing(inside[1], outside[1]);
              indices.push(a, b, d, a, d, c);
            }
          }
        }
      }
    }
    const mesh = {
      positions: Float32Array.from(positions),
      normals: Float32Array.from(normals),
      colors: Float32Array.from(colors),
      indices: Uint32Array.from(indices),
    };
    mesh.vertexCount = mesh.positions.length / 3;
    mesh.triangleCount = mesh.indices.length / 3;
    return fixWinding(mesh);
  }
}

const CUBE_EDGES = [
  [0, 1], [2, 3], [4, 5], [6, 7], // along i
  [0, 2], [1, 3], [4, 6], [5, 7], // along j
  [0, 4], [1, 5], [2, 6], [3, 7], // along k
];

/** Flip triangles whose geometric normal disagrees with the vertex (gradient) normals. */
function fixWinding(mesh) {
  const { positions: p, normals: nrm, indices } = mesh;
  let flipped = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
    const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const dot = nx * (nrm[a * 3] + nrm[b * 3] + nrm[c * 3]) + ny * (nrm[a * 3 + 1] + nrm[b * 3 + 1] + nrm[c * 3 + 1]) + nz * (nrm[a * 3 + 2] + nrm[b * 3 + 2] + nrm[c * 3 + 2]);
    if (dot < 0) {
      indices[t + 1] = c;
      indices[t + 2] = b;
      flipped++;
    }
  }
  mesh.flipped = flipped;
  return mesh;
}

// ---------------------------------------------------------------- cleanup / stats

/** Keep only the connected component (by shared vertices) with most triangles. */
export function largestComponent(mesh) {
  const nv = mesh.positions.length / 3;
  const parent = new Int32Array(nv);
  for (let i = 0; i < nv; i++) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const union = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[a] = b; };
  const idx = mesh.indices;
  for (let t = 0; t < idx.length; t += 3) { union(idx[t], idx[t + 1]); union(idx[t + 1], idx[t + 2]); }
  const triCount = new Map();
  for (let t = 0; t < idx.length; t += 3) { const r = find(idx[t]); triCount.set(r, (triCount.get(r) || 0) + 1); }
  let best = -1, bestN = -1;
  for (const [r, c] of triCount) if (c > bestN) { best = r; bestN = c; }
  const components = triCount.size;
  if (components <= 1) return { ...mesh, components, removedTriangles: 0 };
  const remap = new Int32Array(nv).fill(-1);
  const positions = [], normals = [], colors = [];
  const indices = [];
  for (let t = 0; t < idx.length; t += 3) {
    if (find(idx[t]) !== best) continue;
    for (let s = 0; s < 3; s++) {
      const v = idx[t + s];
      if (remap[v] < 0) {
        remap[v] = positions.length / 3;
        positions.push(mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2]);
        normals.push(mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]);
        colors.push(mesh.colors[v * 3], mesh.colors[v * 3 + 1], mesh.colors[v * 3 + 2]);
      }
      indices.push(remap[v]);
    }
  }
  const out = { positions: Float32Array.from(positions), normals: Float32Array.from(normals), colors: Float32Array.from(colors), indices: Uint32Array.from(indices) };
  out.vertexCount = out.positions.length / 3;
  out.triangleCount = out.indices.length / 3;
  out.components = components;
  out.removedTriangles = mesh.indices.length / 3 - out.triangleCount;
  return out;
}

/** Bounding box, centroid, mean/max distance from the centroid and Euler characteristic (2 = closed sphere-like). */
export function meshStats(mesh) {
  const p = mesh.positions;
  const nv = p.length / 3;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const c = [0, 0, 0];
  for (let i = 0; i < nv; i++) {
    for (let a = 0; a < 3; a++) {
      const v = p[i * 3 + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
      c[a] += v;
    }
  }
  for (let a = 0; a < 3; a++) c[a] /= nv || 1;
  let mean = 0, maxR = 0, minR = Infinity;
  for (let i = 0; i < nv; i++) {
    const r = Math.hypot(p[i * 3] - c[0], p[i * 3 + 1] - c[1], p[i * 3 + 2] - c[2]);
    mean += r;
    if (r > maxR) maxR = r;
    if (r < minR) minR = r;
  }
  const edges = new Set();
  const idx = mesh.indices;
  for (let t = 0; t < idx.length; t += 3) {
    for (let s = 0; s < 3; s++) {
      const a = idx[t + s], b = idx[t + ((s + 1) % 3)];
      edges.add(a < b ? a * nv + b : b * nv + a);
    }
  }
  return { vertices: nv, triangles: idx.length / 3, bbox: { min, max }, centroid: c, meanRadius: mean / (nv || 1), minRadius: minR, maxRadius: maxR, euler: nv - edges.size + idx.length / 3 };
}

/**
 * Analytic depth map of a sphere for tests: returns {depth, alpha} for a camera.
 */
export function sphereDepthMap(cam, center, radius) {
  const { width, height, fx, fy, cx, cy, view: m } = cam;
  const depth = new Float32Array(width * height);
  const alpha = new Float32Array(width * height);
  // sphere centre in camera space
  const sx = m[0] * center[0] + m[4] * center[1] + m[8] * center[2] + m[12];
  const sy = m[1] * center[0] + m[5] * center[1] + m[9] * center[2] + m[13];
  const sz = m[2] * center[0] + m[6] * center[1] + m[10] * center[2] + m[14];
  for (let v = 0; v < height; v++) {
    for (let u = 0; u < width; u++) {
      // ray direction in camera space through the pixel centre
      const dx = (u + 0.5 - cx) / fx, dy = -(v + 0.5 - cy) / fy, dz = -1;
      const dl = Math.hypot(dx, dy, dz);
      const rx = dx / dl, ry = dy / dl, rz = dz / dl;
      const b = rx * sx + ry * sy + rz * sz;
      const c = sx * sx + sy * sy + sz * sz - radius * radius;
      const disc = b * b - c;
      if (disc < 0) continue;
      const t = b - Math.sqrt(disc);
      if (t <= 0) continue;
      depth[v * width + u] = -(rz * t); // view-space z of the hit (positive)
      alpha[v * width + u] = 1;
    }
  }
  return { depth, alpha, width, height };
}
