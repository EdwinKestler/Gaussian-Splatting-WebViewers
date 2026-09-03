/**
 * Unit tests for shared/tsdf.js (plan F6: TSDF fusion + surface nets) and
 * shared/glb.js on an analytic sphere. Run: npm test.
 */
import { test, describe } from "node:test";
import assert from "node:assert/strict";

import { TsdfVolume, fibonacciDirections, largestComponent, lookAtMatrix, meshStats, orbitCameras, sphereDepthMap } from "../../shared/tsdf.js";
import { decodeGlbHeader, encodeGlb } from "../../shared/glb.js";

describe("cameras", () => {
  test("lookAtMatrix maps the target onto the -z axis and Fibonacci directions are unit", () => {
    const m = lookAtMatrix([0, 0, 5], [0, 0, 0]);
    const p = [m[12], m[13], m[14]];
    assert.ok(Math.abs(p[0]) < 1e-6 && Math.abs(p[1]) < 1e-6 && Math.abs(p[2] + 5) < 1e-6, `target → ${p}`);
    for (const d of fibonacciDirections(50)) assert.ok(Math.abs(Math.hypot(...d) - 1) < 1e-9);
    const cams = orbitCameras({ center: [1, 2, 3], distance: 4, count: 6, width: 64, height: 32 });
    assert.equal(cams.length, 6);
    for (const c of cams) {
      assert.ok(Math.abs(Math.hypot(c.eye[0] - 1, c.eye[1] - 2, c.eye[2] - 3) - 4) < 1e-6);
      // the centre projects to the principal point
      const m2 = c.view;
      const z = -(m2[2] * 1 + m2[6] * 2 + m2[10] * 3 + m2[14]);
      const x = m2[0] * 1 + m2[4] * 2 + m2[8] * 3 + m2[12];
      assert.ok(z > 3.99 && Math.abs(x) < 1e-5);
    }
  });
});

describe("TSDF sphere", () => {
  const center = [0.3, -0.2, 0.1];
  const radius = 0.5;
  const build = ({ resolution = 48, views = 16, carve = true } = {}) => {
    const vol = new TsdfVolume({ center, radius, resolution });
    const cams = orbitCameras({ center, distance: 2.5, count: views, width: 96, height: 96 });
    for (const cam of cams) vol.integrate({ ...sphereDepthMap(cam, center, radius), view: cam.view, fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy }, { carve });
    return { vol, mesh: vol.extract() };
  };

  test("fused sphere comes back as a closed mesh with the right radius and outward normals", () => {
    const { vol, mesh } = build();
    assert.equal(vol.views, 16);
    const stats = meshStats(mesh);
    assert.ok(stats.vertices > 500, `vértices ${stats.vertices}`);
    assert.ok(Math.abs(stats.meanRadius - radius) / radius < 0.03, `radio medio ${stats.meanRadius}`);
    assert.ok(stats.maxRadius < radius * 1.12 && stats.minRadius > radius * 0.88, `radios ${stats.minRadius}–${stats.maxRadius}`);
    for (let a = 0; a < 3; a++) assert.ok(Math.abs(stats.centroid[a] - center[a]) < 0.02);
    assert.equal(stats.euler, 2, "esfera cerrada: V − E + F = 2");
    // normals point away from the centre
    let outward = 0;
    for (let i = 0; i < stats.vertices; i++) {
      const dx = mesh.positions[i * 3] - center[0], dy = mesh.positions[i * 3 + 1] - center[1], dz = mesh.positions[i * 3 + 2] - center[2];
      if (dx * mesh.normals[i * 3] + dy * mesh.normals[i * 3 + 1] + dz * mesh.normals[i * 3 + 2] > 0) outward++;
    }
    assert.ok(outward / stats.vertices > 0.98, `normales hacia fuera ${outward}/${stats.vertices}`);
    // triangle winding agrees with the normals (fixWinding left nothing to flip, or flipped consistently)
    const p = mesh.positions, idx = mesh.indices;
    let agree = 0;
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      const ux = p[b * 3] - p[a * 3], uy = p[b * 3 + 1] - p[a * 3 + 1], uz = p[b * 3 + 2] - p[a * 3 + 2];
      const vx = p[c * 3] - p[a * 3], vy = p[c * 3 + 1] - p[a * 3 + 1], vz = p[c * 3 + 2] - p[a * 3 + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const cx = (p[a * 3] + p[b * 3] + p[c * 3]) / 3 - center[0], cy = (p[a * 3 + 1] + p[b * 3 + 1] + p[c * 3 + 1]) / 3 - center[1], cz = (p[a * 3 + 2] + p[b * 3 + 2] + p[c * 3 + 2]) / 3 - center[2];
      if (nx * cx + ny * cy + nz * cz > 0) agree++;
    }
    assert.ok(agree / (idx.length / 3) > 0.98, `orientación ${agree}/${idx.length / 3}`);
    assert.deepEqual(largestComponent(mesh).components, 1);
    assert.ok(mesh.flipped / mesh.triangleCount < 0.01, `orientación base coherente (${mesh.flipped} de ${mesh.triangleCount} corregidas)`);
  });

  test("largestComponent drops a detached fragment", () => {
    const { mesh } = build({ resolution: 32, views: 8 });
    // append a floating triangle far away
    const positions = new Float32Array(mesh.positions.length + 9);
    positions.set(mesh.positions);
    positions.set([9, 9, 9, 9.1, 9, 9, 9, 9.1, 9], mesh.positions.length);
    const normals = new Float32Array(positions.length);
    normals.set(mesh.normals);
    const colors = new Float32Array(positions.length);
    colors.set(mesh.colors);
    const nv = mesh.positions.length / 3;
    const indices = new Uint32Array(mesh.indices.length + 3);
    indices.set(mesh.indices);
    indices.set([nv, nv + 1, nv + 2], mesh.indices.length);
    const out = largestComponent({ positions, normals, colors, indices });
    assert.equal(out.components, 2);
    assert.equal(out.removedTriangles, 1);
    assert.equal(out.triangleCount, mesh.indices.length / 3);
    assert.equal(out.vertexCount, nv);
  });

  test("colour is carried from the views", () => {
    const vol = new TsdfVolume({ center, radius, resolution: 24 });
    const cams = orbitCameras({ center, distance: 2.5, count: 6, width: 48, height: 48 });
    for (const cam of cams) {
      const d = sphereDepthMap(cam, center, radius);
      const color = new Uint8ClampedArray(48 * 48 * 4);
      for (let p = 0; p < 48 * 48; p++) { color[p * 4] = 255; color[p * 4 + 1] = 128; color[p * 4 + 2] = 0; color[p * 4 + 3] = 255; }
      vol.integrate({ ...d, color, view: cam.view, fx: cam.fx, fy: cam.fy, cx: cam.cx, cy: cam.cy });
    }
    const mesh = vol.extract();
    assert.ok(mesh.vertexCount > 50);
    for (let i = 0; i < mesh.vertexCount; i++) {
      assert.ok(Math.abs(mesh.colors[i * 3] - 1) < 0.02 && Math.abs(mesh.colors[i * 3 + 1] - 128 / 255) < 0.02 && mesh.colors[i * 3 + 2] < 0.02);
    }
  });
});

describe("GLB", () => {
  test("encodeGlb writes a valid header, JSON chunk and 4-byte aligned BIN chunk", () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const colors = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const indices = new Uint32Array([0, 1, 2]);
    const glb = encodeGlb({ positions, normals, colors, indices }, { name: "tri", extras: { id_instancia: 3 } });
    assert.equal(glb.byteLength % 4, 0);
    const h = decodeGlbHeader(glb);
    assert.equal(h.version, 2);
    assert.equal(h.length, glb.byteLength);
    assert.equal(h.json.asset.version, "2.0");
    assert.equal(h.json.meshes[0].name, "tri");
    assert.equal(h.json.extras.id_instancia, 3);
    const prim = h.json.meshes[0].primitives[0];
    assert.deepEqual(Object.keys(prim.attributes).sort(), ["COLOR_0", "NORMAL", "POSITION"]);
    const pos = h.json.accessors[prim.attributes.POSITION];
    assert.deepEqual(pos.min, [0, 0, 0]);
    assert.deepEqual(pos.max, [1, 1, 0]);
    assert.equal(h.json.accessors[prim.indices].componentType, 5125);
    assert.equal(h.json.accessors[prim.indices].count, 3);
    assert.equal(h.json.buffers[0].byteLength, h.binLength);
    assert.equal(h.binLength, 3 * 12 * 3 + 12); // 3 float VEC3 views + 3 uint32 indices
    // positions round-trip from the BIN chunk
    const view = h.json.bufferViews[pos.bufferView];
    const back = new Float32Array(glb, h.binOffset + view.byteOffset, 9);
    assert.deepEqual(Array.from(back), Array.from(positions));
    assert.throws(() => encodeGlb({ positions, indices: new Uint32Array([0, 1, 7]) }), /fuera de rango/);
  });
});
